# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample, fix this if its wrong
    if len(preds.shape) == 1:
        _top_max_k_vals, top_max_k_inds = torch.topk(
            preds, min(max(ks), len(labels)), dim=1, largest=True, sorted=True # since we only have 4 labels we can't do top 5 error
        )
    else:
        _top_max_k_vals, top_max_k_inds = torch.topk(
            preds, min(max(ks), len(preds[0])), dim=1, largest=True, sorted=True # since we only have 4 labels we can't do top 5 error
        )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def convert_preds(preds, thresholds=[(0, 3.6), (3.6, 7.4), (7.4, 10)], trg_classes=4):
    """
    Converts a continuous number output to a class label with
    naive rounding (closest class label)
    """
    lower_bounds = torch.tensor([threshold[0] for threshold in thresholds])
    upper_bounds = torch.tensor([threshold[1] for threshold in thresholds])
    class_labels = torch.zeros_like(preds)
    
    for i in range(len(thresholds)):
        if i < len(thresholds) - 1:
            # Condition for all but the last threshold range
            mask = (preds >= lower_bounds[i]) & (preds < upper_bounds[i])
        else:
            # Condition for the last threshold range (inclusive upper bound)
            mask = (preds >= lower_bounds[i]) & (preds <= upper_bounds[i])

        # Assign the class label where the condition is satisfied
        class_labels[mask] = i

    # class_labels = class_labels.unsqueeze(1).to(preds.device)
    class_labels = class_labels.to(preds.device)
    return class_labels

    # class_labels = []
    # for value in preds:
    #     for i, (lower, upper) in enumerate(thresholds):
    #         if i < len(thresholds) - 1:
    #             if lower <= value < upper:
    #                 class_labels.append(i)
    #                 break
    #         else:
    #             if lower <= value <= upper:
    #                 class_labels.append(i)
    #                 break
    # class_labels = torch.tensor(class_labels, device=preds.device)
    # return class_labels.unsqueeze(1)

    # # convert preds from continuous -> class labels
    # indices = torch.floor(preds).long()
    # indices = torch.clamp(indices, 0, trg_classes - 1)
    # return indices.unsqueeze(1)


def compute_continuous(preds):
    num_classes = preds.shape[1]
    classes = np.array([i for i in range(num_classes)])
    intervals = {val: (max(0, val - 0.5), min(num_classes - 1, val + 0.5)) for val in range(num_classes)}
    probabilities = np.array(torch.softmax(preds, dim=1))
    max_indices = np.array(torch.argmax(preds, dim=1))

    final_vals = []
    for cur_element, max_index in enumerate(max_indices):
        interval_size = intervals[max_index][1] - intervals[max_index][0]
        interval_percentage = 1 - probabilities[cur_element][max_index]

        if interval_percentage <= 0.5:
            maximum_contribution = interval_size * 2 * interval_percentage
        elif interval_percentage > 0.5:
            maximum_contribution = interval_size * interval_percentage

        mask = [i != max_index for i in range(num_classes)]
        contribution_percentages = probabilities[cur_element][mask] / np.sum(probabilities[cur_element][mask])
        deltas = classes[mask] - max_index
        signs = np.sign(deltas)
        contribution_limits = maximum_contribution * 1 / 2 ** ((num_classes - 1) - np.abs(classes[mask] - max_index))
        totals = contribution_limits * contribution_percentages * signs
        value = max_index + np.sum(totals)
        final_vals.append(value)

    final_vals = torch.tensor(final_vals, device=preds.device).unsqueeze(1)
    return final_vals


def convert_labels(labels, thresholds=[0, 3.6, 7.4, 10]):
    label_centers = [(thresholds[i] + thresholds[i - 1]) / 2 for i in range(1, len(thresholds))]
    centered_labels = []
    for label in labels:
        centered_labels.append(label_centers[label]) 
    centered_labels = torch.tensor(centered_labels, device=labels.device)
    return centered_labels

    
