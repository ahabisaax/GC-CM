import joblib
from pathlib import Path
from mergedeep import merge
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
)

from xai_concept_leakage.metrics.accs import compute_accuracy
from xai_concept_leakage.models.construction import (
    load_config,
    external_load_model_trainer,
)
from xai_concept_leakage.train.utils import extract_dims
from xai_concept_leakage.train.evaluate import evaluate_representation_metrics
import xai_concept_leakage.interventions.utils as intervention_utils
from xai_concept_leakage.metrics.mutual_information import (
    extract_tril,
    matrix_from_tril,
    estimate_MI_interconcept,
    estimate_MI_concepts_task,
    repeat_estimate_MI_interconcept,
    repeat_estimate_MI_concepts_task,
)

from experiments.experiment_utils import (
    save_joblib,
    model_type_from_name,
    wrap_single_array,
)

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

##################################################################################################
### Predict concepts and task labels:
##################################################################################################


def predict_c_y(
    dl,
    model_path,
    x2c_extractor,
    torch_out=True,
    soft_prob_out=False,
    c_sem_out=False,
    vec_emb_out=False,
    config_path=None
):
    """
    Return predicted concepts and task labels on the given dataloader dl by the model specified at model_path.
    Parameters:
    - torch_out: if True, return torch tensors, else return numpy arrays.
    - soft_prob_out: if True, return soft probabilities over concepts, else return hard predictions.
    - c_sem_out: if True, return the semantic representation c_sem of the concepts.
    c_sem is:
        Soft CBM: soft probs over concepts, same as c_pred: (N_pts, k)
        Logit CBM: logits over concepts: (N_pts, k)
        CEM: concatenation of the weighted sum of c_pos and c_neg as per c_pred: (N_pt, k*emb_size)
    - vec_emb_out: if True, return the concept embeddings c_pos and c_neg.
    """
    model, trainer, config = external_load_model_trainer(
        dl, model_path, x2c_extractor, output_config=True,
        config_path=config_path
    )
    model.eval()
    model_type_vec = config["architecture"] in [
        "ConceptEmbeddingModel",
        "CEM",
        "IntAwareConceptEmbeddingModel",
        "IntCEM",
        "CriticRegularisedConceptEmbeddingModel"
    ]
    with torch.no_grad():
        if model_type_vec:
            predictions = [
                model._forward(
                    batch[0],
                    c=None,
                    y=None,
                    train=False,
                    output_embeddings=vec_emb_out,
                    output_latent=False
                )
                for batch in dl
            ]
        else:
            predictions = [
                model._forward(
                    batch[0],
                    c=None,
                    y=None,
                    train=False,
                )
                for batch in dl
            ]

    c_pred = torch.cat([prediction[0] for prediction in predictions])
    if not soft_prob_out:
        c_pred = (c_pred > 0.5).float()
    y_pred = torch.cat([prediction[2] for prediction in predictions])
    c_true = torch.cat([batch[2] for batch in dl])
    y_true = torch.cat([batch[1] for batch in dl])

    out = [
        c_pred,  # (N_pts, k)
        c_true,  # (N_pts, k)
        y_pred,  # (N_pts, n_classes) in logits
        y_true,
    ]  # (N_pts, 1)

    if c_sem_out:
        c_sem = torch.cat([prediction[1] for prediction in predictions])
        out = [c_sem] + out
    if model_type_vec and vec_emb_out:
        c_pos = torch.cat(
            [prediction[3] for prediction in predictions]
        )  # (N_pts, k, emb_size)
        c_neg = torch.cat(
            [prediction[4] for prediction in predictions]
        )  # (N_pts, k, emb_size)
        out = out + [c_pos, c_neg]
    if not torch_out:
        out = (tens.numpy() for tens in out)
    return out  # c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg


##################################################################################################
### Concept and task accuracies:
##################################################################################################
def compute_concept_task_accuracies(model_path, x2c_extractor, dl_dict, write=False):
    """
    Compute the concept and task accuracies of the model at model_path on the dataloaders in dl_dict.
    Parameters:
    - dl_dict: a dictionary of dataloaders, where the keys are the labels of the dataloaders.
    - write: if True, print the accuracies.
    """
    results = {}
    for dl_label, dl in dl_dict.items():
        results[dl_label] = {}
        (c_pred, c_true, y_pred, y_true) = predict_c_y(
            dl, model_path=model_path, x2c_extractor=x2c_extractor, torch_out=False
        )
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            torch.Tensor(c_pred),
            torch.Tensor(y_pred),
            torch.Tensor(c_true),
            torch.Tensor(y_true),
        )
        y_balanced_accuracy = balanced_accuracy_score(y_true, y_pred.argmax(-1))
        results[dl_label]["y_accuracy"] = y_accuracy
        results[dl_label]["y_balanced_accuracy"] = y_balanced_accuracy
        results[dl_label]["c_accuracy"] = c_accuracy

        if write:
            print(f"Scores of model at {model_path} :")
            print(f"y_accuracy: {y_accuracy:.3f}")
            print(f"y_balanced_accuracy: {y_balanced_accuracy:.3f}")
            print(f"c_accuracy: {c_accuracy:.3f}")
    return results


##################################################################################################
### Leakage scores:
##################################################################################################


def compute_MI_score_model(
    dl,
    model_path,
    x2c_extractor=None,
    score_type="interconcept",
    concept_type="hard",
    wrt_true=False,
    true_scores=None,
    norm_difference=False,
    n_neighbors=3,
    normalise=True,
    n_concepts=None,
):
    """
    Compute the MI score
    - between the concepts and the task labels if score_type = concepts_task,
    - between the concepts if score_type = interconcept.
    Parameters:
    - dl: the dataloader to compute the score on.
    - concept_type: str.
        The type of concepts to use for the score:
        - "true": the true concept labels.
        - "hard": the predicted binary probabilities of the concepts.
        - "soft" or "CEM": the predicted soft probabilities of the concepts.
        - "logit": the predicted concept logits.]
        - "pos"/"neg"/"mix": the predicted positive/negative/weighted embeddings (if a CEM).
    - wrt_true: bool.
        If True, compute the score relative to the true scores.
    - true_scores: the scores computed on the true concept labels.
        If not provided, the true scores are computed.
    - norm_difference: bool.
        If True, return the absolute value of the differences between the score and the true scores.
        If False, return the differences if positive, otherwise 0.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
    - normalise : bool.
        If True, normalises the scores by the concept entropies.
    - n_concepts : int.
        Number of concepts. If None, the number of concepts is inferred from the dataloader.
    """
    if score_type not in ["interconcept", "concepts_task"]:
        raise Exception("score_type must be either 'interconcept' or 'concepts_task'.")
        return None
    if wrt_true and (true_scores is None):
        c_true = torch.cat([batch[2] for batch in dl])
        if score_type == "interconcept":
            true_scores = estimate_MI_interconcept(
                c_true,
                n_neighbors=n_neighbors,
                n_concepts=n_concepts,
                flatten=True,
                normalise=normalise,
            )
        else:
            y_true = torch.cat([batch[1] for batch in dl])
            true_scores = estimate_MI_concepts_task(
                c_true,
                y_true,
                n_neighbors=n_neighbors,
                n_concepts=n_concepts,
                normalise=normalise,
            )
    if concept_type == "true":
        if wrt_true:
            return 0
        else:
            c_true = torch.cat([batch[2] for batch in dl])
            if score_type == "interconcept":
                return estimate_MI_interconcept(
                    c_true,
                    n_neighbors=n_neighbors,
                    flatten=True,
                    n_concepts=n_concepts,
                    normalise=normalise,
                )
            else:
                y_true = torch.cat([batch[1] for batch in dl])
                return estimate_MI_concepts_task(
                    c_true,
                    y_true,
                    n_neighbors=n_neighbors,
                    n_concepts=n_concepts,
                    normalise=normalise,
                )
    else:
        dict_concept_type = {}
        if concept_type in ["pos", "neg"]:
            _, _, _, _, y_true, c_pos, c_neg = predict_c_y(
                dl,
                model_path=model_path,
                x2c_extractor=x2c_extractor,
                c_sem_out=True,
                soft_prob_out=True,
                vec_emb_out=True,
            )
            dict_concept_type.update({"pos": c_pos, "neg": c_neg})
        else:
            c_sem, c_prob, _, _, y_true = predict_c_y(
                dl,
                model_path=model_path,
                x2c_extractor=x2c_extractor,
                c_sem_out=True,
                soft_prob_out=True,
            )
            dict_concept_type.update(
                {
                    "hard": (c_prob > 0.5).float(),
                    "soft": c_prob,
                    "logit": c_sem,
                    "CEM": c_prob,
                    "mix": c_sem,
                }
            )

        if score_type == "interconcept":
            out = estimate_MI_interconcept(
                dict_concept_type[concept_type],
                n_neighbors=n_neighbors,
                flatten=True,
                n_concepts=n_concepts,
                normalise=normalise,
            )
        else:
            out = estimate_MI_concepts_task(
                dict_concept_type[concept_type],
                y_true,
                n_neighbors=n_neighbors,
                n_concepts=n_concepts,
                normalise=normalise,
            )
        if not wrt_true:
            return out
        else:
            out = out - true_scores
            if norm_difference:
                return np.abs(out)
            else:
                return np.maximum(out, 0)


##################################################################################################
### Leakage scores CEMs:
##################################################################################################
def compute_MI_score_CEM(
    cs_n_ys,
    score_type="interconcept",
    vector_type="mix",
    repeats=1,
    n_concepts=None,
    wrt_true=False,
    true_scores=None,
    norm_difference=False,
    normalise=True,
    n_neighbors=3,
):
    """
    Compute the MI score of a CEM on the concept embeddings defined by vector_type
    (either "pos", "neg" or "mix"),
    - between the concept embeddings and the task labels if score_type = concepts_task,
    - between the concept embeddings if score_type = interconcept.
    Unlike compute_MI_score_model, this function takes the outputs of a CEM model on a dataloader
    cs_n_ys = c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg ,
    rather than a dataloader and a model_path.
    Parameters:
    - repeats: int.
        Number of times to repeat the MI estimation.
    - n_concepts : int.
        Number of concepts. Must be provided if vector_type == "mix".
    - wrt_true: bool.
        If True, compute the score relative to the true scores.
    - true_scores: the scores computed on the true concept labels.
        If not provided, the true scores are computed.
    - norm_difference: bool.
        If True, return the absolute value of the differences between the score and the true scores.
        If False, return the differences if positive, otherwise 0.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
    - normalise : bool.
        If True, normalises the scores by the concept entropies.

    """
    if score_type not in ["interconcept", "concepts_task"]:
        raise Exception("score_type must be either 'interconcept' or 'concepts_task'.")
        return None
    c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg = cs_n_ys
    if wrt_true and (true_scores is None):
        if score_type == "interconcept":
            true_scores = estimate_MI_interconcept(
                c_true,
                n_neighbors=n_neighbors,
                n_concepts=n_concepts,
                flatten=True,
                normalise=normalise,
            )
        else:
            true_scores = estimate_MI_concepts_task(
                c_true,
                y_true,
                n_concepts=n_concepts,
                n_neighbors=n_neighbors,
                normalise=normalise,
            )
    dict_vector_types = {
        "mix": c_sem,
        "pos": c_pos,
        "neg": c_neg,
    }
    if score_type == "interconcept":
        out = repeat_estimate_MI_interconcept(
            dict_vector_types[vector_type],
            n_neighbors=n_neighbors,
            flatten=True,
            normalise=normalise,
            n_concepts=n_concepts,
            repeats=repeats,
            return_avg=False,
        )
    else:
        out = repeat_estimate_MI_concepts_task(
            dict_vector_types[vector_type],
            y_true,
            n_neighbors=n_neighbors,
            normalise=normalise,
            n_concepts=n_concepts,
            repeats=repeats,
            return_avg=False,
        )
    if not wrt_true:
        return out
    else:
        out = out - true_scores
        if norm_difference:
            return np.abs(out)
        else:
            return np.maximum(out, 0)


def estimate_MI_cvec_cgt(
    c_vec, c_true, n_concepts, repeats=5, n_neighbors=3, normalise=True
):
    """
    Estimate the MI between the concept vectors c_vec and the true concept labels c_true.
    Return the MI matrix I,
    the average self MI (measuring how predictive on average a concept vector is of its own concept value)
    and the average "other" MI (measuring how predictive on average a concept vector is of the value
    of the other concepts - an indicator of interconcept leakage).
    Parameters:
    - c_vec: the concept vectors to use for the MI estimation.
        Shape: (N_pts, n_concepts, emb_size) or (N_pts, n_concepts * emb_size)
    - c_true: the true concept labels. Shape: (N_pts, n_concepts)
    - n_concepts : int.
        Number of concepts. If None, the number of concepts is inferred from the dataloader.
    - repeats: int.
        Number of times to repeat the MI estimation.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
    - normalise : bool.
        If True, normalises the scores by the concept entropies.
    """
    c_vec = c_vec.reshape(c_vec.shape[0], n_concepts, -1)
    Is = []
    avg_self_MIs = []
    avg_other_MIs = []
    for _ in range(repeats):
        I = np.zeros((n_concepts, n_concepts))
        for ii in range(n_concepts):
            for jj in range(n_concepts):
                I[ii, jj] = estimate_MI_concepts_task(
                    c_vec[:, ii],
                    c_true[:, jj],
                    n_concepts=1,
                    n_neighbors=n_neighbors,
                    normalise=normalise,
                ).item()
        avg_self_MI_vec_c_gt = (np.identity(n_concepts) * I).sum() / n_concepts
        avg_other_MI_vec_c_gt = (I - np.identity(n_concepts) * I).sum() / (
            n_concepts * (n_concepts - 1)
        )
        Is.append(I)
        avg_self_MIs.append(avg_self_MI_vec_c_gt)
        avg_other_MIs.append(avg_other_MI_vec_c_gt)
    return Is, avg_self_MIs, avg_other_MIs


def aligned_unaligned_vectors(c_pos, c_neg, c_true):
    """
    Split the concept vectors into aligned and unaligned vectors.
    Aligned vectors are those that are aligned with the true concept labels
    (i.e. pos if c_true = 1, neg if c_true = 0).
    Return the aligned and unaligned vectors for the positive and negative concepts.
    """
    c_pos_aligned = []
    c_pos_unaligned = []
    c_neg_aligned = []
    c_neg_unaligned = []
    for i_c in range(c_true.shape[1]):
        mask = c_true[:, i_c] > 0.5
        c_pos_aligned.append(c_pos[:, i_c][mask])
        c_pos_unaligned.append(c_pos[:, i_c][~mask])
        c_neg_aligned.append(c_neg[:, i_c][mask])
        c_neg_unaligned.append(c_neg[:, i_c][~mask])
    return c_pos_aligned, c_pos_unaligned, c_neg_aligned, c_neg_unaligned


def aligned_unaligned_MI_score_concept_task_weighted(
    c_pos, c_neg, c_true, y_true, repeats=5, n_neighbors=3, normalise=True
):
    """
    Compute the MI score between the aligned and unaligned concept vectors and the task labels.
    Aligned vectors are those that are aligned with the true concept labels
    (i.e. pos if c_true = 1, neg if c_true = 0).
    It accounts for mild concept imbalance by using the smaller set of aligned vs unaligned for each concept.
    Return a dict with the MI scores for each concept and the average MI scores,
    as well as the per-concept and average scores for alignment leakage.
    Parameters:
    - repeats: int.
        Number of times to repeat the MI estimation.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
    - normalise : bool.
        If True, normalises the scores by the concept entropies.
    """

    scores_per_concept_aligned_pos = []
    scores_per_concept_aligned_neg = []
    scores_per_concept_unaligned_pos = []
    scores_per_concept_unaligned_neg = []

    for i_c in range(c_true.shape[1]):
        cond_smaller_set_inactive = (c_true[:, i_c].mean() > 0.5).item()
        mask_smaller_set = (
            c_true[:, i_c] < 0.5 if cond_smaller_set_inactive else c_true[:, i_c] > 0.5
        )
        n_smaller_set = mask_smaller_set.sum()
        mask_inact = c_true[:, i_c] < 0.5

        y_true_act = y_true[~mask_inact][:n_smaller_set]
        y_true_inact = y_true[mask_inact][:n_smaller_set]

        c_pos_unaligned = c_pos[mask_inact, i_c][:n_smaller_set]
        c_neg_aligned = c_neg[mask_inact, i_c][:n_smaller_set]

        c_pos_aligned = c_pos[~mask_inact, i_c][:n_smaller_set]
        c_neg_unaligned = c_neg[~mask_inact, i_c][:n_smaller_set]

        scores_per_concept_aligned_pos.append(
            repeat_estimate_MI_concepts_task(
                c_pos_aligned,
                y_true_act,
                repeats=repeats,
                n_neighbors=n_neighbors,
                n_concepts=1,
                return_avg=False,
                normalise=normalise,
            )
        )
        scores_per_concept_unaligned_neg.append(
            repeat_estimate_MI_concepts_task(
                c_neg_unaligned,
                y_true_act,
                repeats=repeats,
                n_neighbors=n_neighbors,
                n_concepts=1,
                return_avg=False,
                normalise=normalise,
            )
        )
        scores_per_concept_unaligned_pos.append(
            repeat_estimate_MI_concepts_task(
                c_pos_unaligned,
                y_true_inact,
                repeats=repeats,
                n_neighbors=n_neighbors,
                n_concepts=1,
                return_avg=False,
                normalise=normalise,
            )
        )
        scores_per_concept_aligned_neg.append(
            repeat_estimate_MI_concepts_task(
                c_neg_aligned,
                y_true_inact,
                repeats=repeats,
                n_neighbors=n_neighbors,
                n_concepts=1,
                return_avg=False,
                normalise=normalise,
            )
        )

    scores_per_concept_aligned_pos = reshape_score(scores_per_concept_aligned_pos)
    scores_per_concept_unaligned_pos = reshape_score(scores_per_concept_unaligned_pos)
    scores_per_concept_aligned_neg = reshape_score(scores_per_concept_aligned_neg)
    scores_per_concept_unaligned_neg = reshape_score(scores_per_concept_unaligned_neg)

    out_dict = {
        "CT_MI_i_pos_aligned": scores_per_concept_aligned_pos,
        "CT_MI_i_pos_unaligned": scores_per_concept_unaligned_pos,
        "CT_MI_i_neg_aligned": scores_per_concept_aligned_neg,
        "CT_MI_i_neg_unaligned": scores_per_concept_unaligned_neg,
        "CT_MI_pos_aligned": np.mean(scores_per_concept_aligned_pos, axis=1),
        "CT_MI_pos_unaligned": np.mean(scores_per_concept_unaligned_pos, axis=1),
        "CT_MI_neg_aligned": np.mean(scores_per_concept_aligned_neg, axis=1),
        "CT_MI_neg_unaligned": np.mean(scores_per_concept_unaligned_neg, axis=1),
    }
    out_dict["CT_MI_i_alignment"] = (
        out_dict["CT_MI_i_pos_aligned"]
        - out_dict["CT_MI_i_pos_unaligned"]
        + out_dict["CT_MI_i_neg_aligned"]
        - out_dict["CT_MI_i_neg_unaligned"]
    )
    out_dict["CT_MI_alignment"] = (
        out_dict["CT_MI_pos_aligned"]
        - out_dict["CT_MI_pos_unaligned"]
        + out_dict["CT_MI_neg_aligned"]
        - out_dict["CT_MI_neg_unaligned"]
    )
    return out_dict


def reshape_score(score):
    return np.array([wrap_single_array(arr) for arr in score]).T.squeeze(0)


def aligned_unaligned_MI_score_concept_task_model(
    dl, model_path, x2c_extractor, repeats=5, n_neighbors=3, normalise=True
):
    """
    Wrapper for the function aligned_unaligned_MI_score_concept_task_weighted()
    taking a dataloader and a model path, rather than the concepts and tasks.
    """
    _, c_true, _, y_true, c_pos, c_neg = predict_c_y(
        dl=dl,
        model_path=model_path,
        x2c_extractor=x2c_extractor,
        c_sem_out=False,
        soft_prob_out=True,
        vec_emb_out=True,
    )
    return aligned_unaligned_MI_score_concept_task_weighted(
        c_pos,
        c_neg,
        c_true,
        y_true,
        repeats=repeats,
        n_neighbors=n_neighbors,
        normalise=normalise,
    )


##################################################################################################
### OIS and NIS:
##################################################################################################


def compute_ois_nis_cas(
    model_path,
    dl,
    x2c_extractor,
    train_dl,
    run_ois=False,
    run_nis=False,
    run_cas=False,
    repeats=1,
    rerun=True,
):
    """
    Compute the OIS, NIS and CAS scores of the model at model_path on the dataloader dl.
    Return a dictionary with the scores.
    Adapted from code in the original cem package.
    Parameters:
    - x2c_extractor: the x2c_extractor to use for the model.
    - train_dl: the dataloader for the training set.
    - run_ois: bool. If True, compute the OIS score.
    - run_nis: bool. If True, compute the NIS score.
    - run_cas: bool. If True, compute the CAS score.
    - repeats: int. Number of times to repeat the MI estimation.
    - rerun: bool. If True, forces rerunning the evaluation.
    """
    run_config = load_config(model_path)
    run_config["skip_repr_evaluation"] = False
    run_config["run_ois"] = run_ois
    run_config["run_nis"] = run_nis
    run_config["run_cas"] = run_cas

    list_scores_dict = []
    for repeat in range(repeats):
        scores_dict = evaluate_representation_metrics(
            config=run_config,
            n_concepts=run_config["n_concepts"],
            n_tasks=run_config["n_tasks"],
            test_dl=dl,
            run_name=run_config["run_name"],
            split=run_config["split"],
            imbalance=run_config.get("loss_concept.weight"),
            result_dir=run_config["result_dir"],
            task_class_weights=run_config.get("loss_task.weight"),
            alternative_loading=True,
            train_dl=train_dl,
            model_path=model_path,
            x2c_extractor=x2c_extractor,
            rerun=rerun,
        )
        list_scores_dict.append(scores_dict)
    output_dict = {}
    if run_ois:
        temp_list = [scores_dict["test_ois"] for scores_dict in list_scores_dict]
        output_dict["ois"] = temp_list
    if run_nis:
        temp_list = [scores_dict["test_nis"] for scores_dict in list_scores_dict]
        output_dict["nis"] = temp_list
    if run_cas:
        temp_list = [scores_dict["test_cas"] for scores_dict in list_scores_dict]
        output_dict["cas"] = temp_list
    return output_dict


##################################################################################################
### Interventions:
##################################################################################################
import copy


def run_interventions(
    model_path,
    x2c_extractor,
    train_dl,
    val_dl,
    test_dl,
    competence_level=1,
    policies=["random"],
):
    """
    Run the interventions on the model at model_path on the dataloader test_dl.
    The dataloaders train_dl and val_dl are used to load the model and estimate the policies.
    Adapted from code in the original cem package.
    Parameters:
    - x2c_extractor: the x2c_extractor to use for the model.
    - competence_level: indicates the competence level of the simulated overseer:
        1 means the overseer is perfect, 0 corresponds to adversarial competence.
    - policies: list of str.
        The policies to use for the interventions.
        Allowed names for policy are in the top of xai_concept_leakage/interventions/utils.py:
        examples are "random", "coop", "optimal_greedy".
        If you want to use the learnt intervention policy in IntCEMs,
        then you should also change use_prior to True (see top of interventions/utils.py)
    """
    model, trainer, run_config = external_load_model_trainer(
        train_dl, model_path, x2c_extractor, output_config=True
    )

    intervention_config = copy.deepcopy(run_config["intervention_config"])
    intervention_config["competence_levels"] = [competence_level]
    intervention_config["intervention_policies"] = []
    for policy in policies:
        intervention_config["intervention_policies"].append(
            dict(
                policy=policy,
                use_prior=False,
                group_level=True,
            )
        )
    run_config["intervention_config"] = intervention_config

    concept_map = run_config["concept_map"]
    if concept_map is not None:
        intervened_groups = list(
            range(
                0,
                len(concept_map) + 1,
                intervention_config.get("intervention_freq", 1),
            )
        )
    else:
        intervened_groups = list(
            range(
                0,
                run_config["n_concepts"] + 1,
                intervention_config.get("intervention_freq", 1),
            )
        )

    test_int_args = dict(
        external_loader=True,
        task_class_weights=run_config.get("loss_task.weight"),
        run_name=run_config["run_name"],
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        imbalance=run_config.get("loss_concept.weight"),
        config=run_config,
        n_tasks=run_config["n_tasks"],
        n_concepts=run_config["n_concepts"],
        acquisition_costs=None,
        result_dir=run_config["result_dir"],
        concept_map=run_config["concept_map"],
        intervened_groups=intervened_groups,
        accelerator="auto",
        devices="auto",
        split=run_config["split"],
        rerun=True,
        old_results=None,
        group_level_competencies=intervention_config.get(
            "group_level_competencies",
            False,
        ),
        competence_levels=intervention_config.get(
            "competence_levels",
            [1],
        ),
    )
    if "real_competencies" in intervention_config:
        for real_comp in intervention_config["real_competencies"]:

            def _real_competence_generator(x):
                if real_comp == "same":
                    return x
                if real_comp == "complement":
                    return 1 - x
                if test_int_args["group_level_competencies"]:
                    if real_comp == "unif":
                        batch_group_level_competencies = np.zeros(
                            (x.shape[0], len(concept_map))
                        )
                        for batch_idx in range(x.shape[0]):
                            for group_idx, (_, concept_members) in enumerate(
                                concept_map.items()
                            ):
                                batch_group_level_competencies[
                                    batch_idx,
                                    group_idx,
                                ] = np.random.uniform(
                                    1 / len(concept_members),
                                    1,
                                )
                    else:
                        batch_group_level_competencies = (
                            np.ones((x.shape[0], len(concept_map))) * real_comp
                        )
                    return batch_group_level_competencies

                if real_comp == "unif":
                    return np.random.uniform(
                        0.5,
                        1,
                        size=x.shape,
                    )
                return np.ones(x.shape) * real_comp

            if real_comp == "same":
                # Then we will just run what we normally run
                # as the provided competency matches the level
                # of competency of the user
                test_int_args.pop(
                    "real_competence_generator",
                    None,
                )
                test_int_args.pop(
                    "extra_suffix",
                    None,
                )
                test_int_args.pop(
                    "real_competence_level",
                    None,
                )
            else:
                test_int_args["real_competence_generator"] = _real_competence_generator
                test_int_args["extra_suffix"] = f"_real_comp_{real_comp}_"
                test_int_args["real_competence_level"] = real_comp
    test_results = intervention_utils.test_interventions(**test_int_args)

    model_name = run_config["run_name"] + "_fold_" + str(run_config["split"] + 1)
    return extract_y_acc_interventions_from_output(model_name, test_results, policies)


def run_multiple_interventions(
    model_path,
    x2c_extractor,
    train_dl,
    val_dl,
    test_dl,
    competence_levels=[1],
    policies=["random"],
    repeats=5,
):
    """
    Wrapper for the function run_interventions() that repeats the intervention evaluations
    for multiple competence levels and policies.
    """
    out = {}
    for competence_level in competence_levels:
        full_results = [
            run_interventions(
                model_path,
                x2c_extractor,
                train_dl,
                val_dl,
                test_dl,
                competence_level=competence_level,
                policies=policies,
            )
            for _ in range(repeats)
        ]
        for policy in policies:
            label = policy if competence_level == 1.0 else policy + "_adv"
            out[label] = [test_results[policy] for test_results in full_results]
    return out


def extract_y_acc_interventions_from_output(model_name, test_results, policies):
    """
    Extract the y accuracies from the test results of the interventions.
    """
    out = {
        policy: test_results[key]
        for key in test_results.keys()
        for policy in policies
        if "test_acc_y_" + policy in key
    }
    return out


def plot_pca_structure(data, y_true, save_path, title="PCA Structure by Task Label"):
    """
    Plots the 2D PCA of 'data' (e.g., c_sem or c_pred), colored by 'y_true'.

    Args:
        data: Tensor or Array [Batch_Size, Features]
              (Pass c_sem, c_pred, or flattened c_pos here)
        y_true: Tensor or Array [Batch_Size]
        title: String for the plot title
    """

    # 1. Convert to Numpy (CPU)
    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    X = to_np(data)
    y = to_np(y_true)

    # 2. Handle Dimension Safety
    # If data is 3D (e.g., c_pos [Batch, Concepts, Dim]), flatten it
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    reg_pc1 = LinearRegression().fit(X_pca[:, 0].reshape(-1, 1), y)
    r2_pc1 = reg_pc1.score(X_pca[:, 0].reshape(-1, 1), y)

    reg_plane = LinearRegression().fit(X_pca[:, :2], y)
    r2_pc1_pc2 = reg_plane.score(X_pca[:, :2], y)

    plt.figure(figsize=(10, 7))

    n_classes = len(np.unique(y))
    palette = sns.color_palette("tab10", n_classes) if n_classes <= 10 else "viridis"

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y,
        palette=palette,
        alpha=0.7,
        s=60,
        legend='full'
    )

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    if n_components > 1:
        plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    plt.legend(title="Task Label (y)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return r2_pc1, r2_pc1_pc2

##################################################################################################
### Master results dictionary and single score extraction functions:
##################################################################################################


def evaluate_CBM(
    checkpoint_names,
    results_folder,
    x2c_extractor,
    dl_dict,
    train_dl,
    val_dl,
    test_dl,
    test_dl_label="test",
    list_observables=None,
    ois_repeats=1,
    policies=["random", "coop"],
    intervention_repeats=1,
    n_neighbors=3,
    wrt_true_leakage_scores=True,
    normalise_leakage_scores=True,
    interconcept_repeats=1,
    concepts_task_repeats=1,
    rerun=True,
    save_path=None,
    config_path = None
):
    """
    Master function to evaluate a list of CBM or CEM models in a folder, identified by their checkpoint names.
    Compute the accuracies, OIS, NIS, CAS scores,
    interconcept and concepts-task leakage scores, and intervention performance.
    Save single model results (or update them if they already exist),
    return a dictionary with the overall results, and optionally save the overall results dict to a file.
    Concept and task accuracies are computed on the train, val and test dataloaders.
    The remaining scores are computed on the test dataloader only.
    Parameters:
    - checkpoint_names: list of str. The names of the checkpoints to evaluate
    - results_folder: str. The folder where the checkpoints are stored.
    - x2c_extractor: the x2c_extractor to use for the models.
    - dl_dict: a dictionary of dataloaders, where the keys are the labels of the dataloaders.
        e.g. {"train": train_dl, "val": val_dl, "test": test_dl}.
    - train_dl, val_dl, test_dl: the dataloader for the training, val and test sets.
    - test_dl_label: str. The label of the test dataloader in dl_dict.
    - list_observables: list of str. The observables to compute.
        Allowed values are: "accuracies", "ois", "nis", "cas", "interventions",
        "leakage_scores", "CEM_leakage_scores".
        If None, all observables are computed.
    - ois_repeats: int. Number of times to repeat the OIS and NIS estimation.
    - policies: list of str. The policies to use for the interventions.
        Allowed names for policy are in the top of xai_concept_leakage/interventions/utils.py
    - intervention_repeats: int. Number of times to repeat the intervention evaluation.
    - n_neighbors: int. Number of nearest neighbors to use for the MI estimation.
    - wrt_true_leakage_scores: bool. If True, compute the leakage scores relative to the true scores.
    - normalise_leakage_scores: bool. If True, normalises the leakage scores by the concept entropies.
    - interconcept_repeats: int. Number of times to repeat the interconcept MI estimation.
    - concepts_task_repeats: int. Number of times to repeat the concepts-task MI estimation.
    - rerun: bool. If True, forces rerunning the evaluation.
    - save_path: str. The path to save the overall results to. If None, the overall results are not saved.
    """
    _, n_concepts, _ = extract_dims(train_dl)
    if list_observables is None:
        list_observables = [
            "accuracies",
            "ois",
            "nis",
            "cas",
            "interventions",
            "leakage_scores",
            "CEM_leakage_scores",
        ]
    results = {}

    for checkpoint_name in checkpoint_names:
        print(checkpoint_name)
        checkpoint_path = results_folder + checkpoint_name
        if 'CRCEM' in checkpoint_path:
            model_type = 'CRCEM'
        elif 'CEM' in checkpoint_path:
            model_type = 'CEM'
        elif 'Hard' in checkpoint_path:
            model_type = 'hard'
        else:
            model_type = 'soft'

        match = re.search(r'lam_c([\d\.]+)', checkpoint_path)
        lambda_val = float(match.group(1))
        print(f'lambda_val :{lambda_val}')
        print(model_type)

        results_model = {}
        # Accuracies:
        if "accuracies" in list_observables:
            acc_dict = compute_concept_task_accuracies(
                checkpoint_path,
                x2c_extractor,
                dl_dict,
            )
            merge(results_model, acc_dict)

        # OIS, NIS, CAS scores:
        if any(x in list_observables for x in ["ois", "nis", "cas"]):
            ois_nis_scores_dict = compute_ois_nis_cas(
                checkpoint_path,
                test_dl,
                x2c_extractor,
                train_dl,
                run_ois=("ois" in list_observables),
                run_nis=("nis" in list_observables),
                run_cas=False,
                repeats=ois_repeats,
                rerun=rerun,
            )
            cas_scores_dict = compute_ois_nis_cas(
                checkpoint_path,
                test_dl,
                x2c_extractor,
                train_dl,
                run_ois=False,
                run_nis=False,
                run_cas=("cas" in list_observables),
                repeats=1,
                rerun=rerun,
            )
            merge(results_model, {test_dl_label: ois_nis_scores_dict})
            merge(results_model, {test_dl_label: cas_scores_dict})

        # Intervention accuracies:
        if "interventions" in list_observables:
            interventions_dict = run_multiple_interventions(
                checkpoint_path,
                x2c_extractor,
                train_dl,
                val_dl,
                test_dl,
                policies=policies,
                competence_levels=[0, 1],
                repeats=intervention_repeats,
            )
            merge(results_model, {test_dl_label: interventions_dict})

        # Interconcept and concepts-task leakage scores:
        if "leakage_scores" in list_observables:
            # We don't compute the true scores once and for all folds, as we want to capture the stochasticity
            # in the MI score estimation for the ground-truth too.
            ICL_ij_tril = [
                compute_MI_score_model(
                    dl=dl_dict[test_dl_label],
                    model_path=checkpoint_path,
                    x2c_extractor=x2c_extractor,
                    score_type="interconcept",
                    concept_type=model_type,
                    wrt_true=wrt_true_leakage_scores,
                    normalise=normalise_leakage_scores,
                    n_neighbors=n_neighbors,
                    n_concepts=n_concepts,
                )
                for _ in range(interconcept_repeats)
            ]
            ICL_i = [
                matrix_from_tril(tril).sum(axis=1) / (n_concepts - 1)
                for tril in ICL_ij_tril
            ]
            ICL = [icl_i.sum() / len(icl_i) for icl_i in ICL_i]

            CTL_i = [
                compute_MI_score_model(
                    dl=dl_dict[test_dl_label],
                    model_path=checkpoint_path,
                    x2c_extractor=x2c_extractor,
                    score_type="concepts_task",
                    concept_type=model_type,
                    wrt_true=wrt_true_leakage_scores,
                    normalise=normalise_leakage_scores,
                    n_neighbors=n_neighbors,
                    n_concepts=n_concepts,
                )
                for _ in range(concepts_task_repeats)
            ]
            CTL = [ctl_i.sum() / len(ctl_i) for ctl_i in CTL_i]

            leakage_scores_dict = {
                "ICL_ij_tril": ICL_ij_tril,
                "ICL_i": ICL_i,
                "ICL": ICL,
                "CTL_i": CTL_i,
                "CTL": CTL,
            }
            merge(results_model, {test_dl_label: leakage_scores_dict})

        # Leakage scores for CEMs:
        if (model_type in ["CEM", 'CRCEM']) and ("CEM_leakage_scores" in list_observables):
            c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg = predict_c_y(
                dl=test_dl,
                model_path=checkpoint_path,
                x2c_extractor=x2c_extractor,
                c_sem_out=True,
                soft_prob_out=True,
                vec_emb_out=True,
                config_path=config_path
            )

            # Compute interconcept and concepts-task MIs on predicted concept representations pos, neg, mix:
            for suffix in ["mix", "pos", "neg"]:
                ICL_ij_tril = compute_MI_score_CEM(
                    [c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg],
                    score_type="interconcept",
                    vector_type=suffix,
                    repeats=interconcept_repeats,
                    n_concepts=n_concepts,
                    wrt_true=False,
                    normalise=normalise_leakage_scores,
                    n_neighbors=n_neighbors,
                )
                ICL_i = [
                    matrix_from_tril(tril).sum(axis=1) / (n_concepts - 1)
                    for tril in wrap_single_array(ICL_ij_tril)
                ]
                ICL = [icl_i.sum() / len(icl_i) for icl_i in ICL_i]

                CTL_i = compute_MI_score_CEM(
                    [c_sem, c_pred, c_true, y_pred, y_true, c_pos, c_neg],
                    score_type="concepts_task",
                    vector_type=suffix,
                    repeats=interconcept_repeats,
                    n_concepts=n_concepts,
                    wrt_true=False,
                    normalise=normalise_leakage_scores,
                    n_neighbors=n_neighbors,
                )
                CTL = [ctl_i.sum() / len(ctl_i) for ctl_i in wrap_single_array(CTL_i)]

                dict_vector_types = {"mix": c_sem, "pos": c_pos, "neg": c_neg}

                c_real = dict_vector_types[suffix]
                if hasattr(c_real, 'cpu'):
                    c_real = c_real.cpu().numpy()

                perm_idx = torch.randperm(len(y_true))
                y_shuffled = y_true[perm_idx] if isinstance(y_true, torch.Tensor) else y_true[perm_idx.numpy()]
                baseline_MI_i = compute_MI_score_CEM(
                    [c_real, c_pred, c_true, y_pred, y_shuffled, c_real, c_real],
                    score_type="concepts_task",
                    vector_type=suffix,
                    repeats=interconcept_repeats,
                    n_concepts=n_concepts,
                    wrt_true=False,
                    normalise=normalise_leakage_scores,
                    n_neighbors=n_neighbors,
                )
                # Compute adjusted MI by subtracting baseline bias
                CTL_i_adjusted = [CTL_i[i] - baseline_MI_i[i] for i in range(len(CTL_i))]
                CTL_adjusted = [ctl_i.sum() / len(ctl_i) for ctl_i in wrap_single_array(CTL_i_adjusted)]
                
                # Print comparison of adjusted vs unadjusted MI
                avg_unadjusted = np.mean(CTL)
                avg_adjusted = np.mean(CTL_adjusted)
                avg_diff = avg_unadjusted - avg_adjusted
                print(f"  {suffix}: Unadjusted MI = {avg_unadjusted:.4f}, Adjusted MI = {avg_adjusted:.4f}, Avg Diff = {avg_diff:.4f}")
                
                if suffix == 'mix':
                    emb_size = int(c_sem.shape[1]/n_concepts)
                    c_sem_reshaped = c_sem.view(-1, n_concepts, emb_size)
                    concept_vectors = c_sem_reshaped[:, 0, :]
                    title = f'PCA Structure of {model_type} with with $\lambda_c={lambda_val}$'

                    clean_title = title.translate(str.maketrans({' ': '_', '$': '', '\\': '', '=': ''}))
                    base_filename = f"{clean_title}"

                    # Check slots 1 through 4
                    for idx in range(1, 5):
                        png_path = f"{base_filename}_run{idx}.png"
                        if not os.path.exists(png_path):
                            r2_pc1, r2_pc1_pc2 = plot_pca_structure(data=concept_vectors, y_true=y_true, title=title, save_path=png_path)
                            print(f"Saved to {png_path}")
                            pca_dict = {
                                'R2_pca_1': r2_pc1,
                                'R2_plane': r2_pc1_pc2}
                            merge(results_model, {test_dl_label: pca_dict})
                            break

                leakage_scores_dict = {
                    "IC_MI_ij_tril" + "_" + suffix: ICL_ij_tril,
                    "IC_MI_i" + "_" + suffix: ICL_i,
                    "IC_MI" + "_" + suffix: ICL,
                    "CT_MI_i" + "_" + suffix: CTL_i,
                    "CT_MI" + "_" + suffix: CTL,
                    "CT_MI_i_adjusted" + "_" + suffix: CTL_i_adjusted,
                    "CT_MI_adjusted" + "_" + suffix: CTL_adjusted,
                    "baseline_MI_i" + "_" + suffix: baseline_MI_i,
                }
                merge(results_model, {test_dl_label: leakage_scores_dict})

            # Compute the MI between pos, neg and mix vectors and the true concept labels:
            MI_cvec_cgt_dict = {}
            out = estimate_MI_cvec_cgt(
                c_pos,
                c_true,
                n_concepts,
                repeats=concepts_task_repeats,
                n_neighbors=n_neighbors,
                normalise=normalise_leakage_scores,
            )
            MI_cvec_cgt_dict.update(
                {
                    "MI_pos_c_gt": out[0],
                    "avg_self_MI_pos_c_gt": out[1],
                    "avg_other_MI_pos_c_gt": out[2],
                }
            )
            out = estimate_MI_cvec_cgt(
                c_neg,
                c_true,
                n_concepts,
                repeats=concepts_task_repeats,
                n_neighbors=n_neighbors,
                normalise=normalise_leakage_scores,
            )
            MI_cvec_cgt_dict.update(
                {
                    "MI_neg_c_gt": out[0],
                    "avg_self_MI_neg_c_gt": out[1],
                    "avg_other_MI_neg_c_gt": out[2],
                }
            )
            out = estimate_MI_cvec_cgt(
                c_sem,
                c_true,
                n_concepts,
                repeats=concepts_task_repeats,
                n_neighbors=n_neighbors,
                normalise=normalise_leakage_scores,
            )
            MI_cvec_cgt_dict.update(
                {
                    "MI_mix_c_gt": out[0],
                    "avg_self_MI_mix_c_gt": out[1],
                    "avg_other_MI_mix_c_gt": out[2],
                }
            )
            merge(results_model, {test_dl_label: MI_cvec_cgt_dict})

            # Linear probes: per-concept analysis with train/test split to prevent overfitting
            c_sem_train, c_pred_train, c_true_train, y_pred_train, y_true_train, c_pos_train, c_neg_train = predict_c_y(
                dl=train_dl,
                model_path=checkpoint_path,
                x2c_extractor=x2c_extractor,
                c_sem_out=True,
                soft_prob_out=True,
                vec_emb_out=True,
                config_path=config_path
            )

            z_mix_train = c_sem_train.cpu().numpy() if torch.is_tensor(c_sem_train) else c_sem_train
            y_labels_train = y_true_train.cpu().numpy().ravel() if torch.is_tensor(y_true_train) else y_true_train.ravel()
            c_labels_train = c_true_train.cpu().numpy() if torch.is_tensor(c_true_train) else c_true_train
            c_pred_train = c_pred_train.cpu().numpy() if torch.is_tensor(c_pred_train) else c_pred_train

            z_mix_test = c_sem.cpu().numpy() if torch.is_tensor(c_sem) else c_sem
            y_labels_test = y_true.cpu().numpy().ravel() if torch.is_tensor(y_true) else y_true.ravel()
            c_labels_test = c_true.cpu().numpy() if torch.is_tensor(c_true) else c_true
            c_pred_test = c_pred.cpu().numpy() if torch.is_tensor(c_pred) else c_pred

            emb_size = int(z_mix_train.shape[1] / n_concepts)
            z_mix_train_reshaped = z_mix_train.reshape(-1, n_concepts, emb_size)
            z_mix_test_reshaped = z_mix_test.reshape(-1, n_concepts, emb_size)
            
            # Per-concept probe for task prediction (Y) with L2 regularization
            linear_probe_y_per_concept = []
            for i_c in range(n_concepts):
                z_c_train = z_mix_train_reshaped[:, i_c, :]
                z_c_test = z_mix_test_reshaped[:, i_c, :]
                lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)  # Default L2 regularization
                lr_model.fit(z_c_train, y_labels_train)
                acc_y = lr_model.score(z_c_test, y_labels_test)
                linear_probe_y_per_concept.append(acc_y)
            
            linear_probe_y_avg = np.mean(linear_probe_y_per_concept)


            linear_probe_all_concepts_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            linear_probe_all_concepts_model.fit(z_mix_train, y_labels_train)
            all_concepts_accuracy = linear_probe_all_concepts_model.score(z_mix_test, y_labels_test)

            c_train_hard = (c_pred_train > 0.5).astype(int)
            c_test_hard = (c_pred_test > 0.5).astype(int)
            linear_probe_hard = LogisticRegression(max_iter=1000, random_state=42)
            linear_probe_hard.fit(c_train_hard, y_labels_train)
            binarised_concept_accuracy = linear_probe_hard.score(c_test_hard, y_labels_test)

            
            # Per-concept probe for concept prediction (self and others)
            linear_probe_c_self = []
            linear_probe_c_others_per_concept = []
            
            for i_c in range(n_concepts):
                z_c_train = z_mix_train_reshaped[:, i_c, :]
                z_c_test = z_mix_test_reshaped[:, i_c, :]
                
                # Predict own concept (self)
                lr_self = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                lr_self.fit(z_c_train, c_labels_train[:, i_c])
                acc_self = lr_self.score(z_c_test, c_labels_test[:, i_c])
                linear_probe_c_self.append(acc_self)
                
                # Predict other concepts
                acc_others = []
                for j_c in range(n_concepts):
                    if j_c != i_c:
                        lr_other = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                        lr_other.fit(z_c_train, c_labels_train[:, j_c])
                        acc_other = lr_other.score(z_c_test, c_labels_test[:, j_c])
                        acc_others.append(acc_other)
                
                linear_probe_c_others_per_concept.append(np.mean(acc_others))
            
            linear_probe_c_self_avg = np.mean(linear_probe_c_self)
            linear_probe_c_others_avg = np.mean(linear_probe_c_others_per_concept)

            # Print results
            print(f"  Linear Probe Y per concept: {[f'{acc:.3f}' for acc in linear_probe_y_per_concept]}, Avg: {linear_probe_y_avg:.4f}")
            print(f"  Linear Probe C (self) per concept: {[f'{acc:.3f}' for acc in linear_probe_c_self]}, Avg: {linear_probe_c_self_avg:.4f}")
            print(f"  Linear Probe C (others) per concept: {[f'{acc:.3f}' for acc in linear_probe_c_others_per_concept]}, Avg: {linear_probe_c_others_avg:.4f}")

            linear_probe_dict = {
                "linear_probe_y_per_concept": linear_probe_y_per_concept,
                "linear_probe_y_avg": linear_probe_y_avg,
                "linear_probe_c_self_per_concept": linear_probe_c_self,
                "linear_probe_c_self_avg": linear_probe_c_self_avg,
                "linear_probe_c_others_per_concept": linear_probe_c_others_per_concept,
                "linear_probe_c_others_avg": linear_probe_c_others_avg,
                'linear_probe_y_all_concept': all_concepts_accuracy,
                'linear_probe_binarised_concept': binarised_concept_accuracy}
            merge(results_model, {test_dl_label: linear_probe_dict})

            # Assess alignment leakage:
            aligned_unaligned_CTL_dict = (
                aligned_unaligned_MI_score_concept_task_weighted(
                    c_pos,
                    c_neg,
                    c_true,
                    y_true,
                    repeats=concepts_task_repeats,
                    n_neighbors=n_neighbors,
                    normalise=normalise_leakage_scores,
                )
            )
            merge(results_model, {test_dl_label: aligned_unaligned_CTL_dict})

        # Save single model results (or update them if they already exist):
        results_model_save_path = results_folder + checkpoint_name.split('.pt')[0] + ".dict"
        if Path(results_model_save_path).is_file():
            old_results_model = joblib.load(results_model_save_path)
            results_model = merge({}, old_results_model, results_model)
        joblib.dump(results_model, results_model_save_path)
        merge(results, {checkpoint_name: results_model})
    save_joblib(results, save_path)
    return results
