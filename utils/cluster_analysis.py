"""
Cluster and data analysis functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014, 2021
"""

from sklearn import metrics
from collections import Counter
import numpy as np


def analyse_clusters(labels_true, labels_pred, labels_select=None):
    """
    Analyse clusters and return a list of dict describing the clusters.

    The labels are given as lists. Every dict item in the returned list
    corresponds to a cluster, with the dict having keys "id", "indices",
    "counts", "size", "purity" describing that cluster. If `labels_select` is
    provided, only clusters containing at least one token with a true label in
    labels_select are considered.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    n_clusters = len(set(labels_pred)) - 1
    clusters = []
    for i in set(labels_pred):
        cluster = {}
        cluster["id"] = i  # the index of the cluster

        # Find indices and true labels
        cluster["indices"] = list(np.where(labels_pred == i)[0])
        labels_true_in_cluster = labels_true[cluster["indices"]]
        if labels_select is not None and (
                len(set(labels_true_in_cluster).intersection(
                set(labels_select))) == 0
                ):
            continue

        # Find counts for each of the true labels
        cluster["size"] = 0
        cluster["counts"] = {}
        counts = []
        for label in set(labels_true_in_cluster):
            count = list(labels_true_in_cluster).count(label)
            if labels_select is None or label in labels_select:
                counts.append(count)
            cluster["counts"][label] = count
            cluster["size"] += count

        # Calculuate purity
        if len(counts) == 0:
            cluster["purity"] = 1.0
        else:
            cluster["purity"] = float(max(counts))/cluster["size"]

        clusters.append(cluster)

    return clusters


def purity(labels_true, labels_pred, labels_select=None):
    """
    Calculate and return cluster purity.

    Labels are given as lists. See Section 3 in (Rosenberg and Hirschberg,
    2007) for a formal definition of purity. If `labels_select` is provided,
    only clusters containing at least one token with a true label in
    `labels_select` are considered. Normalization is over the total number of
    tokens in these clusters. See Notebook, 2014-01-30 for the selective case.
    """

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    n_clusters = len(set(labels_pred)) - 1
    n_tokens = 0
    purity = 0.0
    for k in range(n_clusters + 1):

        # Find true labels
        labels_true_in_cluster = labels_true[np.where(labels_pred == k)[0]]
        if labels_select is not None and (
                len(set(labels_true_in_cluster).intersection(
                set(labels_select))) == 0):
            continue

        # Find counts
        counts = []
        for i in set(labels_true_in_cluster):
            count = list(labels_true_in_cluster).count(i)
            if labels_select is None or i in labels_select:
                counts.append(count)
            n_tokens += count

        # Add maximum count to purity
        if counts != []:
            purity += max(counts)

    return purity/n_tokens



def many_to_one_mapping(labels_true, labels_pred):
    """
    Calculate the many-to-one mapping.

    This is the mapping that is essentially used to calculate cluster purity.
    """

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    cluster_to_label_map_many = {}
    for k in sorted(set(labels_pred)):

        # Find true labels
        labels_true_in_cluster = labels_true[np.where(labels_pred == k)[0]]

        # Find counts
        counter = Counter(labels_true_in_cluster)
        cluster_to_label_map_many[k] = counter.most_common(1)[0][0]

    return cluster_to_label_map_many


def one_to_one_mapping(labels_true, labels_pred):
    """
    Calculate the one-to-one mapping and one-to-one accuracy.

    Labels are given as lists. Details of this metric is given in
    (Christodoulopoulos et al., 2009) and (Haghighi and Klein, 2006). Return
    values are the accuracy (float) and a cluster-to-label mapping dict, with
    the keys in the dict the cluster indices (matching indices in the list
    returned by `analyse_clusters`) and the values the label.
    """

    # Construct cluster analysis
    clusters = analyse_clusters(labels_true, labels_pred)

    n_correct = 0
    cluster_to_label_map = {}

    # Greedily map each cluster to a label
    while True:

        # Determine the best cluster-to-label mapping
        i_cluster_to_map = -1
        n_correct_best = 0
        label_to_map = 0
        for i_cluster in range(len(clusters)):

            # Don't consider clusters that have already been mapped
            if clusters[i_cluster]["id"] in cluster_to_label_map:
                continue

            cluster = clusters[i_cluster]

            # Don't consider labels that have already been mapped
            cur_counts = dict(cluster["counts"].items())
            for label in cluster_to_label_map.values():
                if label in cur_counts:
                    cur_counts.pop(label)

            if len(cur_counts) == 0:
                continue

            label_to_map_cur = max(cur_counts, key=lambda x: cur_counts[x])
            n_correct_cur = cur_counts[label_to_map_cur]
            if n_correct_cur > n_correct_best:
                n_correct_best = n_correct_cur
                i_cluster_to_map = clusters[i_cluster]["id"]
                label_to_map = label_to_map_cur

        if i_cluster_to_map == -1:
            break

        n_correct += n_correct_best
        cluster_to_label_map[i_cluster_to_map] = label_to_map

    return float(n_correct)/len(labels_true), cluster_to_label_map


def one_to_one_accuracy(labels_true, labels_pred, labels_select=None):
    """
    Calculate the one-to-one mapping accuracy.

    This function is the same as `one_to_one_mapping`, but it only calculates
    the accuracy, which makes it a lot faster. If `labels_select` is provided,
    only clusters containing at least one token with a true label in
    `labels_select` are considered and mapping is only done to the true labels
    in `labels_select`.
    """

    # Construct cluster analysis
    clusters = analyse_clusters(labels_true, labels_pred, labels_select)

    n_correct = 0
    n_tokens = sum([i["size"] for i in clusters])

    i_mapped = 0

    # Greedily map each cluster to a label
    while True:

        # Determine the best cluster-to-label mapping
        i_cluster_to_map = -1
        n_correct_best = 0
        label_to_map = 0
        for i_cluster in range(len(clusters)):

            cluster = clusters[i_cluster]
            cur_counts = dict(cluster["counts"].items())
            if labels_select is not None:
                new_cur_counts = {}
                for label in cur_counts:
                    if label in labels_select:
                        new_cur_counts[label] = cur_counts[label]
                cur_counts = new_cur_counts
            if len(cur_counts) == 0:
                continue

            label_to_map_cur = max(cur_counts, key=lambda x: cur_counts[x])
            n_correct_cur = cur_counts[label_to_map_cur]
            if n_correct_cur > n_correct_best:
                n_correct_best = n_correct_cur
                i_cluster_to_map = i_cluster
                label_to_map = label_to_map_cur

        if i_cluster_to_map == -1:
            break

        n_correct += n_correct_best

        # Cluster has been mapped, so don't have to consider again
        clusters.pop(i_cluster_to_map)
        i_mapped += 1

        # Label has been mapped, so don't have to consider again
        for cluster in clusters:
            if label_to_map in cluster["counts"]:
                cluster["counts"].pop(label_to_map)
        clusters = [i for i in clusters if len(i["counts"]) != 0]

    return float(n_correct)/n_tokens


def adjusted_rand_score_select(labels_true, labels_pred, labels_select):
    """
    Calculate ARI after removing all tokens not in `labels_select`.

    Only tokens with a true label in `labels_select` are considered in the
    calculation of the adjusted rand index. See Notebook, 2014-01-30.
    """
    new_labels_true = []
    new_labels_pred = []

    for i in range(len(labels_true)):
        if labels_true[i] in labels_select:
            new_labels_true.append(labels_true[i])
            new_labels_pred.append(labels_pred[i])

    return metrics.adjusted_rand_score(new_labels_true, new_labels_pred)


def v_measure_select(labels_true, labels_pred, labels_select):
    """
    Calculate V-measure after removing all tokens not in `labels_select`.

    Only tokens with a true label in `labels_select` are considered in the
    calculation of the V-measure.
    """
    new_labels_true = []
    new_labels_pred = []

    for i in range(len(labels_true)):
        if labels_true[i] in labels_select:
            new_labels_true.append(labels_true[i])
            new_labels_pred.append(labels_pred[i])

    h, c, V = metrics.homogeneity_completeness_v_measure(
        new_labels_true, new_labels_pred)

    return V


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    # Illustrate many-to-one mapping
    labels_true = [
        "apple", "pear", "apple", "apple", "grape", "melon", "pear", "orange",
        "banana", "pear", "pear", "pear", "pear", "orange", "banana", "melon"
        ]
    labels_pred = [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 2]
    expected_mapping_many = {0: "apple", 1: "pear", 2: "pear"}
    mapping_many = many_to_one_mapping(labels_true, labels_pred)
    print("Many-to-one mapping:", mapping_many)
    assert expected_mapping_many == mapping_many

    # Illustrate one-to-one mapping and accuracy
    labels_true = [0, 4, 4, 0, 2, 1, 4, 3, 5, 4, 4, 4, 4, 3, 5, 1]
    labels_pred = [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 2]
    accuracy, mapping = one_to_one_mapping(labels_true, labels_pred)
    print("True labels:", labels_true)
    print("Predicted labels:", labels_pred)
    print("One-to-one mapping:", mapping)
    print("One-to-one mapping accuracy:", accuracy)
    print("Purity:", purity(labels_true, labels_pred))
    assert accuracy == one_to_one_accuracy(labels_true, labels_pred)

    # Illustrate one-to-one accuracy with selected types
    print()
    labels_true = [
        "apple", "pear", "pear", "apple", "grape", "melon", "pear", "orange",
        "banana", "pear", "pear", "pear", "pear", "orange", "banana", "melon"
        ]
    labels_pred = [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 2]
    labels_select = ["apple", "grape", "banana"]
    print("True labels:", labels_true)
    print("Predicted labels:", labels_pred)
    print("Selected labels:", labels_select)
    print(
        "Selective one-to-one mapping accuracy:", one_to_one_accuracy(
        labels_true, labels_pred, labels_select)
        )

    # Illustrate selective purity
    print()
    labels_true = [
        "apple", "pear", "pear", "apple", "grape", "melon", "pear", "orange",
        "banana", "pear", "pear", "pear", "pear", "orange", "banana", "melon"
        ]
    labels_pred = [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 2]
    labels_select = ["apple", "grape", "banana"]
    clusters = analyse_clusters(labels_true, labels_pred, labels_select)
    i = 0
    print("-"*79)
    for cluster in clusters:
        print("Cluster", i)
        for label in cluster["counts"]:
            print('"' + label + '":', cluster["counts"][label])
        print("Size:", cluster["size"])
        print("Purity:", cluster["purity"])
        i += 1
        print("-"*79)
    print("Overall purity:", purity(labels_true, labels_pred))
    print(
        "Selective purity:", purity(
        labels_true, labels_pred, labels_select)
        )
    assert purity(labels_true, labels_pred, labels_select) == (
        (2. + 1) / (6 + 5))
    print("-"*79)


if __name__ == "__main__":
    main()
