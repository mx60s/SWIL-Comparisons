import numpy as np
import pandas as pd
import torch
import os

from copy import copy
from torch import utils
from torchvision.io import read_image

from utils.feature_extractor import *



class SyntheticImageDataset(utils.data.Dataset):
    def __init__(self, annotations_file, class_file, img_dir, transform=None, target_transform=None):
        class_df = pd.read_csv(class_file)
        self.classes = class_df.values.tolist()
        self.annotations = pd.read_csv(annotations_file)
        self.targets = np.array(self.annotations.iloc[:, 1])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path)
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_index_arrays(dataset, indices=[]):
    '''
        Builds numpy arrays of dataset indices corresponding to
        dataset items which should be retained or excluded.

        * USAGE *
        To remove classes from a dataset.

        * PARAMETERS *
        dataset: A torch.utils.data.DataSet ot torchvision.datasets object
        indices: A list of integers representing class indices to remove

        * RETURNS *
        included dataset indices: np.array
        excluded dataset indices: np.array
    '''

    included = []
    excluded = []

    for idx, item in enumerate(dataset):
        if item[1] in indices:
            excluded.append(idx)
        else:
            included.append(idx)

    return np.array(included), np.array(excluded)


def split_training_data(training_data, indices=[]):
    included, excluded = create_index_arrays(training_data, indices)
    incl_set = utils.data.Subset(training_data, included)
    excl_set = utils.data.Subset(training_data, excluded)
    return incl_set, excl_set


def reorder_classes(dataset, new_order):
    '''
        new_order: Dict<k:INT, v:(INT, BOOL)>
            - key: Current class number
            - val[0]: New class number
            - val[1]: Swap T/F
                - T: bidirectional update between two classes ex. 1<-->3
                - F: unidirectional update ex. 1->3 3->4 4->1

        IMPORTANT: Assumes class numbers are 0-indexed
    '''

    old_indices = dict()
    unique_targets = np.unique(dataset.targets).tolist()
    all_targets = np.array(dataset.targets)
    labels = dataset.classes
    new_labels = copy(labels)

    # Gather initial target mappings
    for target in unique_targets:
        old_indices[target] = np.where(all_targets == target)[0]

    for old_target, new_target in new_order.items():
        # Assign new target value to datapoints
        np.put(all_targets, old_indices[old_target], new_target[0])
        # Update label list
        new_labels[new_target[0]] = labels[old_target]

        if new_target[1]: # Swap = True
            # Target and label update if swapping with another class
            np.put(all_targets, old_indices[new_target[0]], old_target)
            new_labels[old_target] = labels[new_target[0]]

    return all_targets.tolist(), new_labels
