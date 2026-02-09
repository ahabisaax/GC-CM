import joblib
import logging
import warnings
import multiprocessing
import numpy as np
import os
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import Callback
import sklearn.metrics
import torch
import torch.nn as nn

from pathlib import Path
from torchvision.models import densenet121

from  xai_concept_leakage.metrics.mutual_information import compute_MI_score_model_training, matrix_from_tril, compute_mi_matrix_parallel


def extract_dims(train_dl):
    x_train, _, c_train = next(iter(train_dl))
    y_train = torch.cat([batch[1] for batch in train_dl])
    n_concepts = c_train.shape[1]
    input_dim = x_train.shape[1]
    if len(y_train.shape) == 1:
        n_tasks = len(torch.unique(y_train))
    else:
        n_tasks = y_train.shape[1]
    return input_dim, n_concepts, n_tasks


def compute_task_class_weights(train_dl):
    _, y_train, _ = next(iter(train_dl))
    y_mean = y_train.type(torch.float32).mean(axis=0).item()
    task_class_weights = torch.Tensor([1 / (1 - y_mean), 1 / y_mean])
    task_class_weights /= torch.min(task_class_weights)
    return task_class_weights


def compute_concept_class_weights(train_dl):
    _, _, c_train = next(iter(train_dl))
    c_mean = c_train.type(torch.float32).mean(axis=0)
    concept_class_weights = 1 / c_mean
    concept_class_weights /= torch.min(concept_class_weights)
    return concept_class_weights


def save_train_val_scores_n_losses(save_path_monitoring, cb_loss):
    print("\nSaving scores and losses to " + save_path_monitoring)
    np.save(save_path_monitoring + "_train_losses", cb_loss.train_losses)
    np.save(save_path_monitoring + "_val_losses", cb_loss.val_losses)
    np.save(save_path_monitoring + "_train_y_acc", cb_loss.train_y_accuracies)
    np.save(save_path_monitoring + "_val_y_acc", cb_loss.val_y_accuracies)
    np.save(save_path_monitoring + "_train_c_acc", cb_loss.train_c_accuracies)
    np.save(save_path_monitoring + "_val_c_acc", cb_loss.val_c_accuracies)
    if not cb_loss.black_box and cb_loss.track_leakage:
        np.save(save_path_monitoring + '_val_ctl', cb_loss.val_ctl)
        np.save(save_path_monitoring + '_val_icl', cb_loss.val_icl)
        np.save(save_path_monitoring + '_norm_val_ctl', cb_loss.val_norm_ctl)
        np.save(save_path_monitoring + '_norm_val_icl', cb_loss.val_norm_icl)
        np.save(save_path_monitoring + '_val_task_icl', cb_loss.val_task_icl)
        np.save(save_path_monitoring + '_val_input_icl', cb_loss.val_input_icl)


def save_train_val_scores_n_losses_indep(
    save_path_monitoring, cb_loss, x2c=False, c2y=False
):
    if x2c:
        print("\nSaving x2c scores and losses to " + save_path_monitoring)
        np.save(save_path_monitoring + "_train_x2c_losses", cb_loss.train_losses)
        np.save(save_path_monitoring + "_val_x2c_losses", cb_loss.val_losses)
        np.save(save_path_monitoring + "_train_c_acc", cb_loss.train_c_accuracies)
        np.save(save_path_monitoring + "_val_c_acc", cb_loss.val_c_accuracies)
    elif c2y:
        print("\nSaving c2y scores and losses to " + save_path_monitoring)
        np.save(save_path_monitoring + "_train_c2y_losses", cb_loss.train_losses)
        np.save(save_path_monitoring + "_val_c2y_losses", cb_loss.val_losses)
        np.save(save_path_monitoring + "_train_y_acc", cb_loss.train_y_accuracies)
        np.save(save_path_monitoring + "_val_y_acc", cb_loss.val_y_accuracies)
    else:
        pass


class LossTracker(Callback):
    def __init__(self,
                 compute_mi_mode='cpu',
                 black_box=False,
                 track_leakage=True, check_leakage=25,
                 every_n_check_val=1):
        super().__init__()
        self.black_box = black_box
        self.track_leakage = track_leakage
        self.check = check_leakage
        self.val_every_n = every_n_check_val
        self.compute_mi_mode = compute_mi_mode

        self.train_loss_temp = []
        self.train_y_accuracy_temp = []
        self.train_losses = []
        self.train_y_accuracies = []

        self.val_loss_temp = []
        self.val_y_accuracy_temp = []
        self.val_losses = []
        self.val_y_accuracies = []

        if not self.black_box:
            self.train_c_accuracy_temp = []
            self.val_c_accuracy_temp = []
            self.train_c_accuracies = []
            self.val_c_accuracies = []

            # needed for CTL and ICL tracking
            if self.track_leakage:
                self.val_c_learnt_temp = []
                self.val_c_true_temp = []
                self.val_y_true_temp = []
                self.val_ctl = []
                self.val_icl= []
                self.val_task_icl = []
                self.val_input_icl = []
                self.val_norm_icl = []
                self.val_norm_ctl = []
                if self.compute_mi_mode == 'both':
                    self.val_c_learnt_temp_gpu = []
                    self.val_c_true_temp_gpu = []
                    self.val_y_true_temp_gpu = []

        self.train_critic_loss_temp = []
        self.train_critic_acc_temp = []
        self.train_critic_losses = []
        self.train_critic_accs = []

    def _avg_of_empty(self, vec):
        if vec == []:
            return 1.0
        else:
            return np.mean(vec)

    @staticmethod
    def compute_leakage_metrics(all_c_learnt,
                                all_c_truth,
                                all_y_true,
                                n_concepts,
                                compute_mi_on_gpu):


        # ICL computation with diagnostics
        mi_icl_learnt = compute_mi_matrix_parallel(
            all_c_learnt,
            d=None,
            n_neighbors=3,
            n_jobs=32,
            max_samples=None,
            normalise=True,
            flatten=True
        )

        mi_icl_true = compute_mi_matrix_parallel(
            all_c_truth,
            d=None,
            n_neighbors=3,
            n_jobs=32,
            max_samples=None,
            normalise=True,
            flatten=True
        )

        # Debug: Check the raw values
        # print("=" * 50)
        # print("ICL DEBUG")
        # print("=" * 50)
        # print(f"mi_icl_learnt shape: {mi_icl_learnt.shape}")
        # print(
        #     f"mi_icl_learnt stats: min={mi_icl_learnt.min():.4f}, max={mi_icl_learnt.max():.4f}, mean={mi_icl_learnt.mean():.4f}")
        # print(f"mi_icl_learnt non-zero: {(mi_icl_learnt > 0).sum()} / {len(mi_icl_learnt)}")
        #
        # print(f"\nmi_icl_true shape: {mi_icl_true.shape}")
        # print(
        #     f"mi_icl_true stats: min={mi_icl_true.min():.4f}, max={mi_icl_true.max():.4f}, mean={mi_icl_true.mean():.4f}")
        # print(f"mi_icl_true non-zero: {(mi_icl_true > 0).sum()} / {len(mi_icl_true)}")
        #
        # # Reconstruct matrices
        mi_matrix_learnt = matrix_from_tril(mi_icl_learnt)
        mi_matrix_true = matrix_from_tril(mi_icl_true)

        print(f"\nReconstructed matrix shapes: {mi_matrix_learnt.shape}, {mi_matrix_true.shape}")
        print(f"Learnt matrix diagonal: {np.diag(mi_matrix_learnt)[:5]}")  # Should be zeros if normalized
        print(f"True matrix diagonal: {np.diag(mi_matrix_true)[:5]}")

        # Per-concept average MI
        per_concept_mi_learnt = mi_matrix_learnt.sum(axis=1) / (n_concepts - 1)
        per_concept_mi_true = mi_matrix_true.sum(axis=1) / (n_concepts - 1)

        print(f"\nPer-concept MI (learnt): {per_concept_mi_learnt[:5]}")
        print(f"Per-concept MI (true): {per_concept_mi_true[:5]}")

        # # Overall averages
        mean_mi_learnt = per_concept_mi_learnt.mean()
        mean_mi_true = per_concept_mi_true.mean()
        mean_norm_icl = np.maximum(0, mean_mi_learnt - mean_mi_true)
        # print(f"Final ICL: {mean_norm_icl:.6f}")
        #
        # # CTL computation with diagnostics
        # print("\n" + "=" * 50)
        # print("CTL DEBUG")
        # print("=" * 50)

        mi_c_learnt_y = compute_mi_matrix_parallel(
            all_c_learnt,
            d=all_y_true,
            n_neighbors=3,
            n_jobs=32,
            max_samples=None,
            normalise=True,
            flatten=False
        )

        mi_c_true_y = compute_mi_matrix_parallel(
            all_c_truth,
            d=all_y_true,
            n_neighbors=3,
            n_jobs=32,
            max_samples=None,
            normalise=True,
            flatten=False
        )

        print(f"mi_c_learnt_y shape: {mi_c_learnt_y.shape}")
        print(
            f"mi_c_learnt_y stats: min={mi_c_learnt_y.min():.4f}, max={mi_c_learnt_y.max():.4f}, mean={mi_c_learnt_y.mean():.4f}")
        print(f"mi_c_learnt_y values: {mi_c_learnt_y[:5]}")

        print(f"\nmi_c_true_y shape: {mi_c_true_y.shape}")
        print(
            f"mi_c_true_y stats: min={mi_c_true_y.min():.4f}, max={mi_c_true_y.max():.4f}, mean={mi_c_true_y.mean():.4f}")
        print(f"mi_c_true_y values: {mi_c_true_y[:5]}")

        mean_mi_c_learnt_y = mi_c_learnt_y.mean()
        mean_mi_c_true_y = mi_c_true_y.mean()

        print(f"\nMean MI(c_learnt, y): {mean_mi_c_learnt_y:.6f}")
        print(f"Mean MI(c_true, y): {mean_mi_c_true_y:.6f}")
        print(f"Difference: {mean_mi_c_learnt_y - mean_mi_c_true_y:.6f}")

        mean_norm_ctl = np.maximum(0, mean_mi_c_learnt_y - mean_mi_c_true_y)
        print(f"Final CTL: {mean_norm_ctl:.6f}")
        #norm_icl = compute_MI_score_model_training(c_pred=all_c_learnt,
        #                                            c_true=all_c_truth,
        #                                            y_true=all_y_true,
        #                                            score_type="interconcept",
        #                                            wrt_true=True,
        #                                            n_neighbors=3,
        #                                            normalise=True,
        #                                            n_concepts=n_concepts,
        #                                            compute_mi_on_gpu=compute_mi_on_gpu
        #                                            )
        # mean_norm_icl_i = matrix_from_tril(norm_icl).sum(axis=1) / (n_concepts - 1)
        # mean_norm_icl = mean_norm_icl_i.sum() / len(mean_norm_icl_i)
        # norm_ctl_vec = compute_MI_score_model_training(c_pred=all_c_learnt,
        #                                                c_true=all_c_truth,
        #                                                y_true=all_y_true,
        #                                                score_type="concepts_task",
        #                                                wrt_true=True,
        #                                                n_neighbors=3,
        #                                                normalise=True,
        #                                                n_concepts=n_concepts,
        #                                                compute_mi_on_gpu=compute_mi_on_gpu)
        #
        # mean_norm_ctl = norm_ctl_vec.sum() / len(norm_ctl_vec)
        return mean_norm_ctl, mean_norm_icl

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, list):
            for output in outputs:
                if output is None:
                    continue

                # 2. Skip if it's not a dictionary
                if not isinstance(output, dict):
                    continue
                is_cbm = "log" in output
                if is_cbm:
                    if "loss" in output:
                        self.train_loss_temp.append(output["loss"].item())

                    # Extract logs
                    logs = output["log"]
                    if "y_accuracy" in logs:
                        self.train_y_accuracy_temp.append(logs["y_accuracy"])
                    if "c_accuracy" in logs:
                        self.train_c_accuracy_temp.append(logs["c_accuracy"])

                else:
                    if "critic_loss" in output:
                        self.train_critic_loss_temp.append(output["critic_loss"].item())
                    elif "loss" in output:
                        self.train_critic_loss_temp.append(output["loss"].item())

                    if "critic_acc" in output:
                        self.train_critic_acc_temp.append(output["critic_acc"])

        elif isinstance(outputs, dict):
            if "loss" in outputs:
                self.train_loss_temp.append(outputs["loss"].item())

            if "log" in outputs:
                logs = outputs["log"]
                if "y_accuracy" in logs:
                    self.train_y_accuracy_temp.append(logs["y_accuracy"])
                if "c_accuracy" in logs:
                    self.train_c_accuracy_temp.append(logs["c_accuracy"])


    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, another_id
    ):
        self.val_loss_temp.append(outputs["val_loss"].item())
        self.val_y_accuracy_temp.append(outputs["val_y_accuracy"])



        if not self.black_box:
            self.val_c_accuracy_temp.append(outputs["val_c_accuracy"])
        if self.track_leakage:
            if "c_learnt" in outputs:
                if self.compute_mi_mode == 'both':
                    self.val_c_learnt_temp_gpu.append(outputs["c_learnt"])
                self.val_c_learnt_temp.append(outputs["c_learnt"])
            if "c_true" in outputs:
                if self.compute_mi_mode == 'both':
                    self.val_c_true_temp_gpu.append(outputs["c_true"])
                self.val_c_true_temp.append(outputs["c_true"])
            if "y_true" in outputs:
                if self.compute_mi_mode == 'both':
                    self.val_y_true_temp_gpu.append(outputs["y_true"])
                self.val_y_true_temp.append(outputs["y_true"])

    def on_train_epoch_end(self, trainer, pl_module):
        #         print("self.train_y_accuracy_temp:")
        #         print(self.train_y_accuracy_temp)
        mean_loss_epoch = self._avg_of_empty(self.train_loss_temp)
        self.train_losses.append(mean_loss_epoch)
        mean_y_accuracy = self._avg_of_empty(self.train_y_accuracy_temp)
        self.train_y_accuracies.append(mean_y_accuracy)
        self.train_loss_temp = []
        self.train_y_accuracy_temp = []
        if not self.black_box:
            mean_c_accuracy = self._avg_of_empty(self.train_c_accuracy_temp)
            self.train_c_accuracies.append(mean_c_accuracy)
            self.train_c_accuracy_temp = []

        self.train_critic_losses.append(self._avg_of_empty(self.train_critic_loss_temp))
        self.train_critic_accs.append(self._avg_of_empty(self.train_critic_acc_temp))
        self.train_critic_loss_temp = []
        self.train_critic_acc_temp = []

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss_epoch = self._avg_of_empty(self.val_loss_temp)
        self.val_losses.append(mean_loss_epoch)
        mean_y_accuracy = self._avg_of_empty(self.val_y_accuracy_temp)
        self.val_y_accuracies.append(mean_y_accuracy)
        self.val_loss_temp = []
        self.val_y_accuracy_temp = []
        pareto_score = 0
        if self.compute_mi_mode == 'cpu':
            device = torch.device('cpu')
        elif self.compute_mi_mode == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                raise RuntimeError("No supported GPU backend found (CUDA or MPS).")

        else:  #both in this case
            device = torch.device('cpu')
            if torch.cuda.is_available():
                device_gpu = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device_gpu = torch.device("mps")
            else:
                raise RuntimeError("No supported GPU backend found (CUDA or MPS).")


        if not self.black_box:
            mean_c_accuracy = self._avg_of_empty(self.val_c_accuracy_temp)
            self.val_c_accuracies.append(mean_c_accuracy)
            self.val_c_accuracy_temp = []

            if self.val_every_n == 1:
                add_1 = 0
            else:
                add_1 = 1
            if self.track_leakage and (
                ((trainer.current_epoch + add_1) % self.check == 0) or
                (trainer.current_epoch == trainer.max_epochs - 1)
            ):

                if self.compute_mi_mode in ['cpu', 'both']:

                    all_c_learnt = torch.cat(self.val_c_learnt_temp).detach().to(device).numpy()
                    all_c_truth = torch.cat(self.val_c_true_temp).detach().to(device).numpy()
                    all_y_true = torch.cat(self.val_y_true_temp).detach().to(device).numpy()

                    n_concepts = all_c_truth.shape[1]
                    ctl, icl = self.compute_leakage_metrics(all_c_learnt=all_c_learnt,
                                                                    all_c_truth=all_c_truth,
                                                                    all_y_true=all_y_true,
                                                                    n_concepts=n_concepts,
                                                                    compute_mi_on_gpu=False)


                    # As binarization occurs, this will look like a 'U' shape
                    if pl_module.logger and hasattr(pl_module.logger, "experiment"):
                        pl_module.logger.experiment.log(
                            {
                                "binarization": wandb.Histogram(all_c_learnt.flatten(),
                                                                num_bins=50)
                            },
                            step=pl_module.current_epoch
                        )

                    # (How close is each concept to its target?)
                    avg_confidence = np.mean(np.abs(all_c_learnt - 0.5) * 2)
                    pl_module.log("stats/binarization_index", avg_confidence)

                    # Visualizes the 'sharpness' of the bottleneck
                    # pl_module.log({
                    #     "binarization/bottleneck_sample": wandb.Image(
                    #         all_c_learnt[:50, :20],
                    #         caption=f"Bottleneck Activations Epoch {trainer.current_epoch}"
                    #     )
                    # })

                    if self.compute_mi_mode == 'both':
                        all_c_learnt_gpu = torch.cat(self.val_c_learnt_temp_gpu).detach().to(device_gpu)
                        all_c_truth_gpu = torch.cat(self.val_c_true_temp_gpu).detach().to(device_gpu)
                        all_y_true_gpu = torch.cat(self.val_y_true_temp_gpu).detach().to(device_gpu)
                        ctl_gpu, icl_gpu = self.compute_leakage_metrics(all_c_learnt=all_c_learnt_gpu,
                                                     all_c_truth=all_c_truth_gpu,
                                                     all_y_true=all_y_true_gpu,
                                                     n_concepts=n_concepts,
                                                     compute_mi_on_gpu=True)
                        pl_module.log('normalised_val_total_icl_gpu', icl_gpu)
                        pl_module.log('val_ctl_normalised_gpu', ctl_gpu)
                        pl_module.log('val_ctl_diff_gpu_cpu', ctl_gpu - ctl)
                        pl_module.log('val_icl_diff_gpu_cpu', icl_gpu - icl)




                else:  #compute mode is gpu
                    all_c_learnt = torch.cat(self.val_c_learnt_temp).detach().to(device_gpu)
                    all_c_truth = torch.cat(self.val_c_true_temp).detach().to(device_gpu)
                    all_y_true = torch.cat(self.val_y_true_temp).detach().to(device_gpu)
                    n_concepts = all_c_truth.shape[1]
                    ctl, icl = self.compute_leakage_metrics(all_c_learnt=all_c_learnt,
                                                                    all_c_truth=all_c_truth,
                                                                    all_y_true=all_y_true,
                                                                    n_concepts=n_concepts,
                                                                    compute_mi_on_gpu=True)



                print(f"ICL: {icl.item():.4f}")
                print(f'CTL: {ctl.item()}')
                self.val_norm_ctl.append(ctl)
                self.val_norm_icl.append(icl)
                pl_module.log('normalised_val_total_icl',icl)
                pl_module.log('val_ctl_normalised', ctl)
                pareto_score = mean_y_accuracy - ctl.item()

            if self.track_leakage:
                self.val_c_learnt_temp.clear()
                self.val_c_true_temp.clear()
                self.val_y_true_temp.clear()
                if self.compute_mi_mode == 'both':
                    self.val_y_true_temp_gpu.clear()
                    self.val_c_true_temp_gpu.clear()
                    self.val_c_learnt_temp_gpu.clear()

        pl_module.log("val_pareto_score", pareto_score)

################################################################################
## HELPER FUNCTIONS
################################################################################

def _save_result(fun, kwargs, output_filepath):
    result = fun(**kwargs)
    joblib.dump(result, output_filepath)
    return result


def execute_and_save(
    fun,
    kwargs,
    result_dir,
    filename,
    rerun=False,
):
    output_filepath = os.path.join(
        result_dir,
        filename,
    )
    if (not rerun) and os.path.exists(output_filepath):
        return joblib.load(output_filepath)
    context = multiprocessing.get_context("spawn")
    p = context.Process(
        target=_save_result,
        kwargs=dict(
            fun=fun,
            kwargs=kwargs,
            output_filepath=output_filepath,
        ),
    )
    p.start()
    p.join()
    if p.exitcode:
        raise ValueError(f"Subprocess failed!")
    p.kill()
    return joblib.load(output_filepath)


def load_call(
    function,
    keys,
    run_name,
    old_results=None,
    rerun=False,
    kwargs=None,
):
    old_results = old_results or {}
    kwargs = kwargs or {}
    if not isinstance(keys, (tuple, list)):
        keys = [keys]

    outputs = []
    for key in keys:
        if key.endswith("_" + run_name):
            real_key = key[: len(run_name) + 1]
        else:
            real_key = key
        rerun = rerun or (
            os.environ.get(f"RERUN_METRIC_{real_key.upper()}", "0") == "1"
        )
        if real_key in old_results:
            outputs.append(old_results[real_key])
        else:
            rerun = True
            logging.debug(
                f"Restarting run because we could not find {real_key} in "
                f"old results for {run_name}."
            )
            break
    if not rerun:
        return outputs, True

    return function(**kwargs), False


def _to_val(x):
    if len(x) >= 2 and (x[0] == "[") and (x[-1] == "]"):
        return eval(x)
    try:
        return int(x)
    except ValueError:
        # Then this is not an int
        pass

    try:
        return float(x)
    except ValueError:
        # Then this is not an float
        pass

    if x.lower().strip() in ["true"]:
        return True
    if x.lower().strip() in ["false"]:
        return False

    return x


def extend_with_global_params(config, global_params):
    for param_path, value in global_params:
        var_names = list(map(lambda x: x.strip(), param_path.split(".")))
        current_obj = config
        for path_entry in var_names[:-1]:
            if path_entry not in config:
                current_obj[path_entry] = {}
            current_obj = current_obj[path_entry]
        current_obj[var_names[-1]] = _to_val(value)


def compute_bin_accuracy(y_pred, y_true):
    y_probs = y_pred.reshape(-1).cpu().detach()
    y_pred = y_probs > 0.5
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class="ovo",
        )
    except:
        y_auc = 0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except:
        y_f1 = 0
    return (y_accuracy, y_auc, y_f1)


def compute_accuracy(
    y_pred,
    y_true,
    binary_output=False,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1) or binary_output:
        return compute_bin_accuracy(
            y_pred=y_pred,
            y_true=y_true,
        )
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class="ovo",
        )
    except:
        y_auc = 0.0
    y_f1 = 0.0
    return (y_accuracy, y_auc, y_f1)


def wrap_pretrained_model(c_extractor_arch, pretrain_model=True):
    def _result_x2c_fun(output_dim):
        try:
            model = c_extractor_arch(pretrained=pretrain_model)
            if output_dim:
                if c_extractor_arch == densenet121:
                    model.classifier = torch.nn.Linear(
                        1024,
                        output_dim,
                    )
                elif hasattr(model, "fc"):
                    model.fc = torch.nn.Linear(512, output_dim)
        except:
            model = c_extractor_arch(
                output_dim=output_dim,
            )
        return model

    return _result_x2c_fun


################################################################################
## HELPER CLASSES
################################################################################


class EmptyEnter(object):
    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return None

    def __exit__(self, *args, **kwargs):
        pass


class ActivationMonitorWrapper:
    def __init__(
        self,
        model,
        trainer,
        activation_freq,
        single_frequency_epochs,
        output_dir,
        test_dl,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.activation_freq = activation_freq
        self.single_frequency_epochs = single_frequency_epochs
        self.output_dir = output_dir
        self.test_dl = test_dl
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.epoch = 0
        self.trainer = trainer
        self.model = model

    @property
    def current_epoch(self):
        return self.trainer.current_epoch

    def fit(self, *args, **kwargs):
        if self.epoch == 0:
            self.dump_activations()
        true_max_epochs = self.trainer.max_epochs
        while self.epoch < true_max_epochs:
            if self.epoch < self.single_frequency_epochs:
                next_size = 1
            else:
                next_size = min(
                    true_max_epochs - self.epoch,
                    self.activation_freq,
                )
            self.trainer.fit_loop.max_epochs = next_size + self.epoch
            self.trainer.fit_loop.current_epoch = self.epoch
            self.trainer.fit(*args, **kwargs)
            self.epoch += next_size
            self.dump_activations()

    def dump_activations(self):
        batch_results = self.trainer.predict(self.model, self.test_dl)
        out_semantics = np.concatenate(
            list(map(lambda x: x[0], batch_results)),
            axis=0,
        )
        out_embs = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )

        out_acts = np.concatenate(
            list(map(lambda x: x[2], batch_results)),
            axis=0,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_embedding_semantics_on_epoch_{self.epoch}.npy",
            ),
            out_semantics,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_embedding_vectors_on_epoch_{self.epoch}.npy",
            ),
            out_embs,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_model_output_on_epoch_{self.epoch}.npy",
            ),
            out_acts,
        )


class WrapperModule(pl.LightningModule):
    def __init__(
        self,
        model,
        n_tasks,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        optimizer="sgd",
        top_k_accuracy=2,
        binary_output=False,
        weight_loss=None,
        sigmoidal_output=False,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.binary_output = binary_output
        self.model = model
        if self.n_tasks > 1 and (not binary_output):
            self.loss_task = torch.nn.CrossEntropyLoss(weight=weight_loss)
        elif not sigmoidal_output:
            self.loss_task = torch.nn.BCEWithLogitsLoss(weight=weight_loss)
        else:
            self.loss_task = torch.nn.BCELoss(weight=weight_loss)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        if (not isinstance(top_k_accuracy, list)) and top_k_accuracy:
            top_k_accuracy = [top_k_accuracy]
        self.top_k_accuracy = top_k_accuracy
        if sigmoidal_output:
            self.sig = torch.nn.Sigmoid()
            self.acc_sig = lambda x: x
        else:
            # Then we assume the model already outputs a sigmoidal vector
            self.sig = lambda x: x
            self.acc_sig = torch.nn.Sigmoid() if self.binary_output else lambda x: x

    def forward(self, x):
        return self.sig(self.model(x))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def _run_step(self, batch, batch_idx, train=False):
        x, y = batch
        y_logits = self(x)
        loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            y,
        )
        # compute accuracy
        (y_accuracy, y_auc, y_f1) = compute_accuracy(
            y_true=y,
            y_pred=self.acc_sig(y_logits),
            binary_output=self.binary_output,
        )

        result = {
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "loss": loss.detach(),
        }
        if (
            (self.top_k_accuracy is not None)
            and (self.n_tasks > 2)
            and (not self.binary_output)
        ):
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                result[f"y_top_{top_k_val}_accuracy"] = y_top_k_accuracy
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))
        return {
            "loss": loss,
            "log": {
                "y_accuracy": result["y_accuracy"],
                "y_auc": result["y_auc"],
                "y_f1": result["y_f1"],
                "loss": result["loss"],
            },
        }

    def validation_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("accuracy" in name))
        return {"val_" + key: val for key, val in result.items()}

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result["loss"]

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
