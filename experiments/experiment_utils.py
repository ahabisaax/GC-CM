import copy
import itertools
import logging
import numpy as np
from scipy.stats import sem
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from collections import defaultdict
from pathlib import Path
from prettytable import PrettyTable

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


################################################################################
## TabularToy:
################################################################################


class MLP_x2c(nn.Module):
    def __init__(
        self,
        input_dim,
        n_concepts,
        hidden_size=512,
        dropout_prob=0.2,
        all_dropout=False,
    ):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_concepts = n_concepts
        self.dropout_prob = dropout_prob
        self.all_dropout = all_dropout
        super().__init__()
        if all_dropout:
            all_dropout = 1.0
        else:
            all_dropout = 0.0

        if type(hidden_size) == list:
            hidden_size = [input_dim] + hidden_size
            lis = [
                Layer(
                    hidden_size[i],
                    hidden_size[i + 1],
                    all_dropout,
                    dropout_prob,
                    LN=True,
                )
                for i in range(len(hidden_size) - 1)
            ]
            lis += [
                Layer(hidden_size[-1], n_concepts, all_dropout, dropout_prob, LN=False)
            ]
            self.MLP = nn.Sequential(*lis)
        else:
            self.MLP = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=all_dropout * dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=all_dropout * dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(p=all_dropout * dropout_prob),
                nn.Linear(hidden_size, n_concepts),
                # nn.LayerNorm(n_concepts),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                #                         nn.Linear(hidden_size, n_tasks),
            )
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.MLP(x)


def get_tabulartoy_extractor_arch(input_dim, config):
    def _c_extractor_arch(output_dim):
        return MLP_x2c(
            input_dim,
            n_concepts=output_dim,
            hidden_size=config["hidden_size"],
            dropout_prob=config["dropout_prob"],
            all_dropout=config["all_dropout"],
        )

    return _c_extractor_arch


################################################################################
## dSprites:
################################################################################
def get_dsprites_extractor_arch():
    one_channel_input = True

    def _c_extractor_arch(output_dim):
        model = resnet18(pretrained=True)
        model = list(model.children())
        model = model[:-1] + [
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=512, out_features=output_dim, bias=True),
        ]
        if one_channel_input:
            w = model[0].weight
            model[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
            model[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        return nn.Sequential(*model)

    return _c_extractor_arch


def get_shapes3d_extractor_arch():
    one_channel_input = False

    def _c_extractor_arch(output_dim):
        model = resnet18(pretrained=True)
        model = list(model.children())
        model = model[:-1] + [
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=512, out_features=output_dim, bias=True),
        ]
        if one_channel_input:
            w = model[0].weight
            model[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
            model[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        return nn.Sequential(*model)

    return _c_extractor_arch


################################################################################
## HELPER FUNCTIONS
################################################################################


def determine_rerun(
    config,
    rerun,
    run_name,
    split,
):
    if rerun:
        return True
    reruns = config.get("reruns", [])
    if "RERUNS" in os.environ:
        reruns += os.environ["RERUNS"].split(",")
    for variant in [
        run_name,
        run_name + f"_split_{split}",
        run_name + f"_fold_{split}",
    ]:
        if variant in reruns:
            return True
    return False


def get_mnist_extractor_arch(input_shape, num_operands):
    def c_extractor_arch(output_dim):
        intermediate_maps = 16
        output_dim = output_dim or 128
        return torch.nn.Sequential(
            *[
                torch.nn.Conv2d(
                    in_channels=num_operands,
                    out_channels=intermediate_maps,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                torch.nn.BatchNorm2d(num_features=intermediate_maps),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(
                    in_channels=intermediate_maps,
                    out_channels=intermediate_maps,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                torch.nn.BatchNorm2d(num_features=intermediate_maps),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(
                    in_channels=intermediate_maps,
                    out_channels=intermediate_maps,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                torch.nn.BatchNorm2d(num_features=intermediate_maps),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(
                    in_channels=intermediate_maps,
                    out_channels=intermediate_maps,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                torch.nn.BatchNorm2d(num_features=intermediate_maps),
                torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(
                    int(np.prod(input_shape[2:])) * intermediate_maps,
                    output_dim,
                ),
            ]
        )

    return c_extractor_arch


def get_metric_from_dict(results, method, metric):
    vals = []
    for _, metric_keys in results.items():
        for candidate_method, metric_map in metric_keys.items():
            if method != candidate_method:
                continue
            for metric_name, val in metric_map.items():
                if metric_name == metric:
                    vals.append(val)
    return vals


def perform_model_selection(
    results,
    model_groupings,
    selection_metric,
    name_filters=None,
):
    name_filters = name_filters or []
    un_select_method = lambda x: np.any([re.search(reg, x) for reg in name_filters])
    new_results = defaultdict(lambda: defaultdict(dict))
    method_names = set()
    for _, metric_keys in results.items():
        for method_name, metric_map in metric_keys.items():
            # Make sure we do not select a method that has been filtered out
            if un_select_method(method_name):
                continue
            method_names.add(method_name)

    selection_result = {}
    for group_pattern, group_name in model_groupings:
        selected_methods = [
            name for name in method_names if re.search(group_pattern, name)
        ]
        selected_values = [
            (
                method_name,
                np.mean(
                    get_metric_from_dict(
                        results,
                        method_name,
                        selection_metric,
                    ),
                ),
            )
            for method_name in selected_methods
        ]
        if selected_values:
            selected_values.sort(key=lambda x: -x[1])
            selected_method = selected_values[0][0]
            group_name = group_name or selected_method
            selection_result[group_name] = selected_method
            for fold, metric_keys in results.items():
                new_results[fold][group_name] = copy.deepcopy(
                    results[fold][selected_method]
                )
    return dict(new_results), selection_result


def perform_averaging(
    results,
    model_groupings,
    name_filters=None,
):
    name_filters = name_filters or []
    un_select_method = lambda x: np.any([re.search(reg, x) for reg in name_filters])
    new_results = defaultdict(lambda: defaultdict(dict))
    method_names = set()
    metric_names = set()
    for _, metric_keys in results.items():
        for method_name, metric_map in metric_keys.items():
            # Make sure we do not select a method that has been filtered out
            if un_select_method(method_name):
                continue
            method_names.add(method_name)
            for metric_name, _ in metric_map.items():
                metric_names.add(metric_name)

    for group_pattern, group_name in model_groupings:
        selected_methods = [
            name for name in method_names if re.search(group_pattern, name)
        ]
        for fold, metric_keys in results.items():
            for metric_name in metric_names:
                avg = None
                count = 0
                for method_name in selected_methods:
                    if not metric_name in results[fold][method_name]:
                        continue
                    if avg is None:
                        avg = results[fold][method_name][metric_name]
                    else:
                        avg += results[fold][method_name][metric_name]
                    count += 1
                if count:
                    new_results[fold][group_name][metric_name] = avg / count
    return new_results


def print_table(
    results,
    result_dir,
    split=0,
    summary_table_metrics=None,
    sort_key="model",
    config=None,
    save_name="output_table",
):
    config = config or {}
    # Initialise output table
    results_table = PrettyTable()
    field_names = [
        "Method",
        "Task Accuracy",
    ]
    result_table_fields_keys = [
        "test_acc_y",
    ]

    # Add AUC only when it is a binary class
    shared_params = config.get("shared_params", {})
    if shared_params.get("n_tasks", 3) <= 2:
        field_names.append("Task AUC")
        result_table_fields_keys.append("test_auc_y")

    # Now add concept evaluation metrics
    field_names.extend(
        [
            "Concept Accuracy",
            "Concept AUC",
        ]
    )
    result_table_fields_keys.extend(
        [
            "test_acc_c",
            "test_auc_c",
        ]
    )

    # CAS, if we chose to compute it (off by default as it may be
    # computationally expensive)
    if (not shared_params.get("skip_repr_evaluation", False)) and shared_params.get(
        "run_cas", True
    ):
        field_names.append("CAS")
        result_table_fields_keys.append("test_cas")

    # And intervention summaries if we chose to also include them
    if (
        len(
            shared_params.get("intervention_config", {}).get(
                "intervention_policies", []
            )
        )
        > 0
    ):
        field_names.extend(
            [
                "25% Int Acc",
                "50% Int Acc",
                "75% Int Acc",
                "100% Int Acc",
            ]
        )
        result_table_fields_keys.extend(
            [
                "test_acc_y_random_group_level_True_use_prior_False_ints_25%",
                "test_acc_y_random_group_level_True_use_prior_False_ints_50%",
                "test_acc_y_random_group_level_True_use_prior_False_ints_75%",
                "test_acc_y_random_group_level_True_use_prior_False_ints_100%",
            ]
        )

    if summary_table_metrics is not None:
        for field in summary_table_metrics:
            if not isinstance(field, (tuple, list)):
                field = field, field
            field_name, field_pretty_name = field
            result_table_fields_keys.append(field_name)
            field_names.append(field_pretty_name)
    results_table.field_names = field_names
    table_rows_inds = {name: i for (i, name) in enumerate(result_table_fields_keys)}
    table_rows = {}
    end_results = defaultdict(lambda: defaultdict(list))
    for fold_idx, metric_keys in results.items():
        for method_name, metric_vals in metric_keys.items():
            for metric_name, vals in metric_vals.items():
                for desired_metric in result_table_fields_keys:
                    real_name = desired_metric
                    if desired_metric.startswith("test_acc_y_") and (
                        ("_ints_" in desired_metric) and (desired_metric[-1] == "%")
                    ):
                        # Then we are dealing with some interventions we wish
                        # to log
                        percent = int(
                            desired_metric[desired_metric.rfind("_") + 1 : -1]
                        )
                        desired_metric = desired_metric[: desired_metric.rfind("_")]
                    else:
                        percent = None

                    if metric_name == desired_metric:
                        if percent is None:
                            end_results[real_name][method_name].append(vals)
                        else:
                            end_results[real_name][method_name].append(
                                vals[int((len(vals) - 1) * percent / 100)]
                            )

    for metric_name, runs in end_results.items():
        for method_name, trial_results in runs.items():
            if method_name not in table_rows:
                table_rows[method_name] = [
                    (None, None) for _ in result_table_fields_keys
                ]
            try:
                (mean, std) = np.mean(trial_results), np.std(trial_results)
                if metric_name in table_rows_inds:
                    table_rows[method_name][table_rows_inds[metric_name]] = (mean, std)
            except:
                logging.warning(
                    f"\tWe could not average results "
                    f"for {metric_name} in model {method_name}"
                )
    table_rows = list(table_rows.items())
    if sort_key == "model":
        # Then sort based on method name
        table_rows.sort(key=lambda x: x[0], reverse=True)
    elif sort_key in table_rows_inds:
        # Else sort based on the requested parameter
        table_rows.sort(
            key=lambda x: (
                x[1][table_rows_inds[sort_key]][0]
                if x[1][table_rows_inds[sort_key]][0] is not None
                else -float("inf")
            ),
            reverse=True,
        )
    for aggr_key, row in table_rows:
        for i, (mean, std) in enumerate(row):
            if mean is None or std is None:
                row[i] = "N/A"
            elif int(mean) == float(mean):
                row[i] = f"{mean} ± {std:}"
            else:
                row[i] = f"{mean:.4f} ± {std:.4f}"
        results_table.add_row([str(aggr_key)] + row)
    print("\t", "*" * 30)
    print(results_table)
    print("\n\n")

    # Also serialize the results
    if result_dir:
        with open(
            os.path.join(result_dir, f"{save_name}_fold_{split + 1}.txt"),
            "w",
        ) as f:
            f.write(str(results_table))


def filter_results(results, run_name, cut=False):
    output = {}
    for key, val in results.items():
        if run_name not in key:
            continue
        if cut:
            key = key[: -len("_" + run_name)]
        output[key] = val
    return output


def evaluate_expressions(config, parent_config=None, soft=False):
    parent_config = parent_config or config
    for key, val in config.items():
        if isinstance(val, (str,)):
            if len(val) >= 4 and (val[0:2] == "{{" and val[-2:] == "}}"):
                # Then do a simple substitution here
                try:
                    config[key] = val[2:-2].format(**parent_config)
                    config[key] = eval(config[key])
                except Exception as e:
                    if soft:
                        # Then we silently ignore this error
                        pass
                    else:
                        # otherwise we just simply raise it again!
                        raise e
            else:
                config[key] = val.format(**parent_config)
        elif isinstance(val, dict):
            # Then we progress recursively
            evaluate_expressions(val, parent_config=parent_config)


def initialize_result_directory(results_dir):
    Path(
        os.path.join(
            results_dir,
            "models",
        )
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            results_dir,
            "history",
        )
    ).mkdir(parents=True, exist_ok=True)


def generate_hyperparameter_configs(config):
    if "grid_variables" not in config:
        # Then nothing to see here so we will return
        # a singleton set with this config in it
        return [config]
    # Else time to do some hyperparameter search in here!
    vars = config["grid_variables"]
    options = []
    for var in vars:
        if var not in config:
            raise ValueError(
                f'All variable names in "grid_variables" must be exhisting '
                f"fields in the config. However, we could not find any field "
                f'with name "{var}".'
            )
        if not isinstance(config[var], list):
            raise ValueError(
                f"If we are doing a hyperparamter search over variable "
                f'"{var}", we expect it to be a list of values. Instead '
                f"we got {config[var]}."
            )
        options.append(config[var])
    mode = config.get("grid_search_mode", "exhaustive").lower().strip()
    if mode in ["grid", "exhaustive"]:
        iterator = itertools.product(*options)
    elif mode in ["paired"]:
        iterator = zip(*options)
    else:
        raise ValueError(
            f"The only supported values for grid_search_mode "
            f'are "paired" and "exhaustive". We got {mode} '
            f"instead."
        )
    result = []
    for specific_vals in iterator:
        current = copy.deepcopy(config)
        for var_name, new_val in zip(vars, specific_vals):
            current[var_name] = new_val
        result.append(current)
    return result


##################################################################################################
### Helper functions:
##################################################################################################
import joblib


def save_joblib(out, save_path):
    if save_path is not None:
        file_name = save_path.split("/")[-1]
        save_folder = save_path[: -len(file_name)]
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        joblib.dump(out, save_path)


from xai_concept_leakage.models.construction import load_config


def model_type_from_name(model_path):
    cfg = load_config(model_path)
    if (cfg["architecture"] == "IndependentConceptBottleneckModel") and (
        cfg["bool"] == True
    ):
        return "hard"
    elif (cfg["architecture"] == "ConceptBottleneckModel") and cfg["sigmoidal_prob"]:
        return "soft"
    elif cfg["architecture"] == "ConceptBottleneckModel":
        return "logit"
    elif cfg["architecture"] == "ConceptEmbeddingModel":
        return "CEM"
    else:
        return "unknown"


def wrap_single_array(arrays):
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    return arrays


##################################################################################################
### Process results and extract scores:
##################################################################################################


def extract_scores_from_results(
    results, test_dl_label="test", score_labels=None, avg=False, checkpoint_names=None
):
    """
    Extract a set of scores from the results dictionary.
    Parameters:
    - test_dl_label: str. Scores are extracted for this dataloaders.
    - score_labels: list of str. The labels of the scores to extract.
    - avg: bool. If True, the scores are averaged across the folds and [avg, SE] are returned.
    - checkpoint_names: list of str. The names of the checkpoints to extract the scores from.
        If None, all checkpoints in the results are taken.
    """
    scores_dict = {}
    if score_labels is None:
        print(f"No input score_label provided.")
        return scores_dict
    else:
        if checkpoint_names == None:
            checkpoint_names = list(results.keys())
        for checkpoint_name in checkpoint_names:
            scores_dict[checkpoint_name] = {}
            for score_label in score_labels:
                vec = np.array(results[checkpoint_name][test_dl_label][score_label])
                if avg:
                    if score_label in ["ICL", "CTL"]:
                        axis = 1
                    else:
                        axis = 0
                    if vec.shape[axis] == 1.0:
                        mean = np.mean(vec, axis=axis)
                        SE = 0.0 * (np.zeros_like(mean) == 0)
                        out = [mean, SE]
                    else:
                        mean, SE = np.mean(vec, axis=axis), sem(vec, axis=axis)
                        out = np.vstack((mean, SE)).T
                        if out.shape[0] == 1:
                            out = out.ravel()
                else:
                    out = vec

                if len(score_labels) == 1:
                    scores_dict[checkpoint_name] = out
                else:
                    scores_dict[checkpoint_name][score_label] = out
        return scores_dict
