from typing import Union, List
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plot_matrix(mat: Union[pd.DataFrame, np.ndarray], fontsz: int, cbar_ticks: List[float] = None, to_show: bool = True):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Othereise, matrix is anonymous
    :param fontsz: font size
    :param cbar_ticks: the spacing between cbar ticks. If None, this is set automatically.
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    if cbar_ticks is not None:
        ax = sns.heatmap(mat, cmap=cmap, vmin=min(cbar_ticks), vmax=max(cbar_ticks), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    else:
        ax = sns.heatmap(mat, cmap=cmap, vmin=np.min(np.array(mat).ravel()), vmax=np.max(np.array(mat).ravel()), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsz)
    if to_show:
        plt.show()
    else:
        plt.close()


def correlation_matrix(df: pd.DataFrame, font_size: int = 10, corrThr: float = None, to_show: bool = True):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param font_size: font size
    :param toShow: True - plots the figure
    :param corrThr: for easy highlight of significant correlations. Above corrThr, consider the threshold = 1.0. This will highlight the correlative pair
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    if corrThr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= corrThr] = 1.0
        corr_mat[corr_mat <= -corrThr] = -1.0

    # print(corr_mat.to_string())

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plot_matrix(corr_mat, fontsz=font_size, cbar_ticks=cbar_ticks, to_show=to_show)


def print_cm(cm: np.ndarray, labels: list, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    labels_as_strings = [str(label) for label in labels]
    columnwidth = max([len(x) for x in labels_as_strings] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels_as_strings:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels_as_strings):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels_as_strings)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def cm_and_classification_report(y: np.ndarray, y_pred: np.ndarray, labels: list):
    cm = confusion_matrix(y, y_pred, labels=labels)
    # Confusion matrix
    print_cm(cm, labels=labels)
    print("")
    # Print the confusion matrix, precision and recall, among other metrics
    print(classification_report(y, y_pred, digits=3))
    print("")



