import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def show_confusion_matrix(matrix, categories, title, fname=None):
    assert isinstance(matrix, np.ndarray) and matrix.ndim == 2

    df_cm = pd.DataFrame(matrix, index=categories, columns=categories)
    plt.figure(figsize=(13, 13))
    g = sn.heatmap(df_cm,
                   annot=True, fmt='.2f',
                   xticklabels=categories, yticklabels=categories,
                   cmap="Blues",
                   vmin=0, vmax=1)

    g.set_xticklabels(categories, rotation=-65, fontsize=13)
    g.set_yticklabels(categories, rotation=-30, fontsize=13)

    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('True', fontsize=13)
    plt.title(title)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight')
    plt.close()


def show_confusion_matrix_with_counts(matrix, counts, categories, title,
                                      fname=None, mean_counts=False, show_subset=True):
    assert isinstance(matrix, np.ndarray) and matrix.ndim == 2
    assert isinstance(counts, np.ndarray) and counts.ndim == 2

    counts = counts.astype(int)

    sorted_categories = sorted(categories)
    inverse_sorted_categories = sorted(categories, reverse=True)
    sorted_indices = [categories.index(c) for c in sorted_categories]

    sorted_counts = [counts[i] for i in sorted_indices]

    df_cm = pd.DataFrame(matrix, index=categories, columns=categories)
    df_cm_sorted = df_cm[sorted_categories]
    df_cm_sorted = df_cm_sorted.sort_index(ascending=True)

    if show_subset:
        df_cm_sorted = df_cm_sorted[df_cm_sorted.sum(1, skipna=True) > 0]
        keep_columns = list(df_cm_sorted.index)
        df_cm_sorted = df_cm_sorted[keep_columns]
        sorted_counts = [sorted_counts[i] for i, c in enumerate(sorted_categories) if c in keep_columns]

    if len(sorted_counts) == 0:
        return
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [12, 1]}, figsize=(13, 12))
    g0 = sn.heatmap(df_cm_sorted,
                    annot=True, fmt='.2f',
                    xticklabels=sorted_categories, yticklabels=sorted_categories,
                    cmap="Blues",
                    cbar=False,
                    vmin=0, vmax=2,
                    ax=a0,
                    annot_kws={"fontsize": 14})
    g0.set_xticklabels(sorted_categories, rotation=-65, fontsize=16)
    g0.set_yticklabels(sorted_categories, rotation=-30, fontsize=16)
    g0.set_xlabel('Predicted', fontsize=16)
    g0.set_ylabel('True', fontsize=16)
    g0.set_title(title, fontsize=15)

    g1 = sn.heatmap(sorted_counts,
                    annot=True, fmt='d',
                    xticklabels=False, yticklabels=False,
                    cmap="Greens",
                    cbar=False,
                    ax=a1,
                    annot_kws={"fontsize": 14})
    g1.set_title('Counts' if mean_counts is False else 'Avg Counts', fontsize=16)

    f.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    # show_confusion_matrix(np.array([[0.1,0.1,0.13],[0,0.4,0],[0,0.8,0]]),
    #                       ['a', 'b', 'c'],
    #                       "cm",
    #                       None)
    show_confusion_matrix_with_counts(np.array([[0.02, 0.85, 0, 0, 0.13, 0, 0],
                                                [0, 0.78, 0.17, 0, 0, 0, 0],
                                                [0,0.76,0.18,0,0,0,0],
                                                [0,0.76,0.18,0,0,0,0],
                                                [0,0.76,0.18,0,0,0,0],
                                                [0,0.76,0.18,0,0,0,0],
                                                [0,0.76,0.18,0,0,0,0],
                                                ]),
                                      np.array([[10],[10],[15],[20],[7],[6],[10]]),
                                      ['Colonial', 'NeoClassicism', 'Ottoman', 'Gothic', 'Byzantine', 'Baroque', 'Romaneswue'],
                                      "Blah\nBli",
                                      "tmp.jpg")
