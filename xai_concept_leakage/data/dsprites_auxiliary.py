"""
Adapted from https://github.com/mateoespinosa/concept-quality
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from xai_concept_leakage.data.utils import logits_from_probs
import xai_concept_leakage.data.dSprites as dsprites
import xai_concept_leakage.data.latentFactorData as latentFactorData


#################################################################
# Select concepts and impose correlations:
#################################################################


def count_class_balance(y):
    one_hot = tf.keras.utils.to_categorical(y)
    return np.sum(one_hot, axis=0) / one_hot.shape[0]


def multiclass_binary_concepts_map_fn(concepts):
    new_concepts = np.zeros((concepts.shape[0], 5))
    # We will have 5 concepts:
    # (0) "is it ellipse or square?"
    new_concepts[:, 0] = (concepts[:, 0] < 2).astype(int)

    # (1) "is_size < 3?"
    num_sizes = len(set(concepts[:, 1]))
    new_concepts[:, 1] = (concepts[:, 1] < num_sizes / 2).astype(int)

    # (2) "is rotation < PI/2?"
    num_rots = len(set(concepts[:, 2]))
    new_concepts[:, 2] = (concepts[:, 2] < num_rots / 2).astype(int)

    # (3) "is x <= 16?"
    num_x_coords = len(set(concepts[:, 3]))
    new_concepts[:, 3] = (concepts[:, 3] < num_x_coords // 2).astype(int)

    # (4) "is y <= 16?"
    num_y_coords = len(set(concepts[:, 4]))
    new_concepts[:, 4] = (concepts[:, 4] < num_y_coords // 2).astype(int)

    return new_concepts


def _get_concept_vector(c_data):
    return np.array(
        [
            # First check if it is an ellipse or a square
            int(c_data[0] < 2),
            # Now check that it is "small"
            int(c_data[1] < 3),
            # And it has not been rotated more than PI/2 radians
            int(c_data[2] < 20),
            # Finally, check whether it is in not in the the upper-left quadrant
            int(c_data[3] < 15),
            int(c_data[4] < 15),
        ]
    )


def binary_from_dec(dec_int):
    out = list(format(dec_int, "08b")[-5:])
    return [int(num) for num in out]


def dec_from_binary(binary_digits):
    return int("".join(list(map(str, binary_digits))), 2)


# Version from the paper:
def multiclass_task_label_fn(c_data):
    concept_vector = _get_concept_vector(c_data)
    if concept_vector[0] == 1:
        return dec_from_binary([concept_vector[1], concept_vector[3]])
    else:
        return 4 + dec_from_binary([concept_vector[2], concept_vector[4]])


# Version from the repo:
# def multiclass_task_label_fn(c_data):
#     # Our task will be a binary task where we are interested in determining
#     # whether an image is a "small" ellipse not in the upper-left
#     # quadrant that has been rotated less than 3*PI/2 radians
#     concept_vector = _get_concept_vector(c_data)
#     binary_label_encoding = [
#         concept_vector[0] or concept_vector[1],
#         concept_vector[2] or concept_vector[3],
#         concept_vector[4],
#     ]
#     return int(
#         "".join(list(map(str, binary_label_encoding))),
#         2
#     )


def dep_0_filter_fn(concept):
    ranges = [
        list(range(3)),
        list(range(0, 6, 2)),
        list(range(0, 40, 4)),
        list(range(0, 32, 2)),
        list(range(0, 32, 2)),
    ]
    return all([(concept[i] in ranges[i]) for i in range(len(ranges))])


scale_shape_sets_lower = [list(np.random.permutation(4))[:3] for i in range(3)]

scale_shape_sets_upper = [list(2 + np.random.permutation(4))[:3] for i in range(3)]


def dep_1_filter_fn(concept):
    ranges = [
        list(range(3)),
        list(range(6)),
        list(range(0, 40, 4)),
        list(range(0, 32, 2)),
        list(range(0, 32, 2)),
    ]

    concept_vector = _get_concept_vector(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False
    if concept_vector[0]:
        if concept[1] not in scale_shape_sets_lower[concept[0]]:
            return False
    else:
        if concept[0] not in scale_shape_sets_upper[concept[0]]:
            return False
    return True


rotation_scale_sets_lower = [list(np.random.permutation(30))[:20] for i in range(6)]

rotation_scale_sets_upper = [
    list(10 + np.random.permutation(30))[:20] for i in range(6)
]


def dep_2_filter_fn(concept):
    ranges = [
        list(range(3)),
        list(range(6)),
        list(range(0, 40, 2)),
        list(range(0, 32, 2)),
        list(range(0, 32, 2)),
    ]

    concept_vector = _get_concept_vector(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False
    if concept_vector[0]:
        if concept[1] not in scale_shape_sets_lower[concept[0]]:
            return False
    else:
        if concept[0] not in scale_shape_sets_upper[concept[0]]:
            return False

    if concept_vector[1]:
        if concept[2] not in rotation_scale_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in rotation_scale_sets_upper[concept[1]]:
            return False
    return True


x_pos_rotation_sets_lower = [list(np.random.permutation(20))[:16] for i in range(40)]

x_pos_rotation_sets_upper = [
    list(12 + np.random.permutation(20))[:16] for i in range(40)
]


def dep_3_filter_fn(concept):
    ranges = [
        list(range(3)),
        list(range(6)),
        list(range(0, 40, 2)),
        list(range(0, 32)),
        list(range(0, 32, 2)),
    ]

    concept_vector = _get_concept_vector(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False
    if concept_vector[0]:
        if concept[1] not in scale_shape_sets_lower[concept[0]]:
            return False
    else:
        if concept[0] not in scale_shape_sets_upper[concept[0]]:
            return False

    if concept_vector[1]:
        if concept[2] not in rotation_scale_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in rotation_scale_sets_upper[concept[1]]:
            return False

    if concept_vector[2]:
        if concept[3] not in x_pos_rotation_sets_lower[concept[2]]:
            return False
    else:
        if concept[3] not in x_pos_rotation_sets_upper[concept[2]]:
            return False
    return True


y_pos_x_pos_sets_lower = [list(np.random.permutation(20))[:16] for i in range(32)]

y_pos_x_pos_sets_upper = [list(12 + np.random.permutation(20))[:16] for i in range(32)]


def dep_4_filter_fn(concept):
    ranges = [
        list(range(3)),
        list(range(6)),
        list(range(0, 40, 2)),
        list(range(0, 32)),
        list(range(0, 32)),
    ]

    concept_vector = _get_concept_vector(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False
    if concept_vector[0]:
        if concept[1] not in scale_shape_sets_lower[concept[0]]:
            return False
    else:
        if concept[0] not in scale_shape_sets_upper[concept[0]]:
            return False

    if concept_vector[1]:
        if concept[2] not in rotation_scale_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in rotation_scale_sets_upper[concept[1]]:
            return False

    if concept_vector[2]:
        if concept[3] not in x_pos_rotation_sets_lower[concept[2]]:
            return False
    else:
        if concept[3] not in x_pos_rotation_sets_upper[concept[2]]:
            return False

    if concept_vector[3]:
        if concept[4] not in y_pos_x_pos_sets_lower[concept[3]]:
            return False
    else:
        if concept[4] not in y_pos_x_pos_sets_upper[concept[3]]:
            return False
    return True


#################################################################
# Define binary task:
#################################################################


def balanced_multiclass_task_label_fn(c_data):
    # Our task will be a binary task where we are interested in determining
    # whether an image is a "small" ellipse not in the upper-left
    # quadrant that has been rotated less than 3*PI/2 radians
    concept_vector = _get_concept_vector(c_data)
    threshold = 0
    if concept_vector[0] == 1:
        binary_label_encoding = [
            concept_vector[1],
            concept_vector[2],
        ]
    else:
        threshold = 4
        binary_label_encoding = [
            concept_vector[3],
            concept_vector[4],
        ]
    return threshold + int("".join(list(map(str, binary_label_encoding))), 2)


# Checked it is equivalent to the original implementation at all levels.
def balanced_multiclass_task_label_fn_equivalent(c_data):
    # Our task will be a binary task where we are interested in determining
    # whether an image is a "small" ellipse not in the upper-left
    # quadrant that has been rotated less than 3*PI/2 radians
    concept_vector = _get_concept_vector(c_data)
    return (1 - concept_vector[0]) * (
        2 * concept_vector[3] + concept_vector[4] + 4
    ) + concept_vector[0] * (2 * concept_vector[1] + concept_vector[2])


def balanced_multiclass_task_label_fn_nonlinear(c_data):
    concept_vector = _get_concept_vector(c_data)
    return (
        (1 - concept_vector[0]) * (2 * concept_vector[3] + concept_vector[4] + 4)
        + concept_vector[0] * (2 * concept_vector[1] + concept_vector[2])
        - concept_vector[0] * concept_vector[1] * concept_vector[4]
        - concept_vector[0] * concept_vector[2] * concept_vector[4]
        - concept_vector[0] * concept_vector[1] * concept_vector[3]
        - (1 - concept_vector[0]) * concept_vector[2] * concept_vector[3]
    )


#################################################################
# Generate dataset:
#################################################################


def generate_dsprites_dataset(
    label_fn,
    filter_fn=None,
    dataset_path=None,
    concept_map_fn=lambda x: x,
    sample_map_fn=lambda x: x,
    dsprites_path="dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    force_reload=False,
):
    if (not force_reload) and dataset_path and os.path.exists(dataset_path):
        # Them time to load up this dataset!
        ds = np.load(dataset_path)
        return (
            (ds["x_train"], ds["y_train"], ds["c_train"]),
            (ds["x_test"], ds["y_test"], ds["c_test"]),
        )

    def _task_fn(x_data, c_data):
        return latentFactorData.get_task_data(
            x_data=x_data,
            c_data=c_data,
            label_fn=label_fn,
            filter_fn=filter_fn,
        )

    loaded_dataset = dsprites.dSprites(
        dataset_path=dsprites_path,
        train_size=0.8,
        random_state=42,
        task=_task_fn,
    )
    _, _, _ = loaded_dataset.load_data()

    x_train = sample_map_fn(loaded_dataset.x_train)
    y_train = loaded_dataset.y_train
    c_train = concept_map_fn(loaded_dataset.c_train)

    x_test = sample_map_fn(loaded_dataset.x_test)
    y_test = loaded_dataset.y_test
    c_test = concept_map_fn(loaded_dataset.c_test)

    if dataset_path:
        # Then serialize it to speed up things next time
        np.savez(
            dataset_path,
            x_train=x_train,
            y_train=y_train,
            c_train=c_train,
            x_test=x_test,
            y_test=y_test,
            c_test=c_test,
        )
    return (
        (x_train, y_train, c_train),
        (x_test, y_test, c_test),
    )


#################################################################
# Create dataloaders:
#################################################################


def dsprites_dataloaders(
    path_dataset,
    val_ratio=0.1,
    num_workers=1,
    batch_size=32,
    considered_concepts=[0, 1, 2, 3, 4],
    c_logits=False,
):
    x_train, y_train, c_train, x_test, y_test, c_test = list(
        np.load(path_dataset).values()
    )
    val_step = int(0.8 / val_ratio)
    val_mask = np.array([not (i % val_step > 0) for i in range(x_train.shape[0])])
    if c_logits:
        c_train, c_test = [logits_from_probs(c) for c in [c_train, c_test]]

    x = torch.FloatTensor(x_train[~val_mask])
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_train[~val_mask])[:, considered_concepts]
    y = torch.LongTensor(y_train[~val_mask])
    train_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers,
        pin_memory=True
    )

    x = torch.FloatTensor(x_train[val_mask])
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_train[val_mask])[:, considered_concepts]
    y = torch.LongTensor(y_train[val_mask])
    val_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers,
        pin_memory=(num_workers > 0), persistent_workers=(num_workers > 0)
    )

    x = torch.FloatTensor(x_test)
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_test)[:, considered_concepts]
    y = torch.LongTensor(y_test)
    test_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers,
        pin_memory=True
    )
    return train_dl, val_dl, test_dl
