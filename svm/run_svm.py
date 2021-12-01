import argparse
import numpy as np
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from svm.eval import f1score
from svm.methods.svm import SVM as SVMSimple
from svm.methods.svm_unique import SVM as SVMunique
from svm.methods.svm_with_curvature import SVM as SVMcurvature
from svm.methods.svm_hog_unique import SVM as SvmHogUnique
from svm.methods.random import RandomPredictor
from svm.methods.random_unique import RandomPredictor as RandomPredictorUnique
from svm.visual import show_confusion_matrix_with_counts

from common.utils import STYLES as classes

IGNORE_CLASSES = []

balanced_accuracy = 0.
conf_mat = np.zeros((len(classes), len(classes)))


def run(data_dir, models_dir, folds, svm_impl="SVM"):

    os.makedirs(models_dir, exist_ok=True)

    stat_metrics = []
    eval_metrics = []
    for classname in classes:
        if classname in IGNORE_CLASSES:
            continue
        for sample in ['c', 'g']:
            for data_split in ['Tr', 'Te']:
                for metric in ['Pct', 'Cnt']:
                    stat_metrics.append('{}{}{}{}'.format(metric, sample, data_split, classname))
    for sample in ['c', 'g']:
        for metric in ['P', 'R']:
            for classname in classes:
                if classname in IGNORE_CLASSES:
                    continue
                eval_metrics.append('{}{}{}'.format(sample, metric, classname))

    confusion_mat = np.zeros((len(classes), len(classes)))
    confusion_mat_per_group = np.zeros((len(classes), len(classes)))
    occurrence_mat = np.zeros_like(confusion_mat)

    stat_metrics_per_split = {'Split': []}
    eval_metrics_per_split = {'Split': []}
    for metric in eval_metrics:
        eval_metrics_per_split.setdefault(metric, [])
    for metric in stat_metrics:
        stat_metrics_per_split.setdefault(metric, [])

    for fold in folds:

        train_fold_file = "{}/trainfold{}.csv".format(data_dir, fold)
        test_fold_file = "{}/testfold{}.csv".format(data_dir, fold)
        if not os.path.exists(train_fold_file) or not os.path.exists(test_fold_file):
            train_fold_file = "{}/fold{}/split_train_test/train.txt".format(data_dir, fold)
            test_fold_file = "{}/fold{}/split_train_test/test.txt".format(data_dir, fold)

        model_pkl_file = "{}/fold{}.pkl".format(models_dir, fold)
        eval_metrics_per_split['Split'].append(fold)
        stat_metrics_per_split['Split'].append(fold)
        if svm_impl == "simple":
            impl = SVMSimple
            named_args = {'ignore_classes': IGNORE_CLASSES}
        elif svm_impl == "unique":
            impl = SVMunique
            named_args = {'ignore_classes': IGNORE_CLASSES,
                          'unique_dirs': args.unique_dirs.split(",")}
        elif svm_impl == "curvature":
            impl = SVMcurvature
            named_args = {'ignore_classes': IGNORE_CLASSES}
        elif svm_impl == "random":  # not svm. just random prediction.
            impl = RandomPredictor
            named_args = {'ignore_classes': IGNORE_CLASSES}
        elif svm_impl == "random_unique":  # not svm. just random prediction.
            impl = RandomPredictorUnique
            named_args = {'ignore_classes': IGNORE_CLASSES,
                          'unique_dirs': args.unique_dirs.split(",")}
        elif svm_impl == "hog_unique":
            impl = SvmHogUnique
            named_args = {'ignore_classes': IGNORE_CLASSES,
                          'unique_dirs': args.unique_dirs.split(",")}
        else:
            raise Exception(f"unknown implementation {svm_impl}")
        result = impl(train_fold_file, test_fold_file, classes, model_pkl_file, **named_args).run()
        if len(result) == 4:
            fold_metrics, cmn, cmn_per_group, occ = result
        else:
            fold_metrics, cmn, occ = result
            cmn_per_group = np.zeros_like(cmn)

        for metric in stat_metrics:
            if metric in fold_metrics:
                stat_metrics_per_split[metric].append(fold_metrics[metric])
            else:
                stat_metrics_per_split[metric].append(np.nan)
        for metric in eval_metrics:
            if metric in fold_metrics:
                eval_metrics_per_split[metric].append(fold_metrics[metric])
            else:
                eval_metrics_per_split[metric].append(np.nan)

        confusion_mat += cmn
        confusion_mat_per_group += cmn_per_group
        occurrence_mat += occ

    df = pd.DataFrame.from_dict(eval_metrics_per_split)
    df.loc['mean'] = df.mean().round(3)
    df.to_csv("{}/eval.csv".format(models_dir), index=True)
    f1score("{}/eval.csv".format(models_dir), avg_f1_nan=eval(args.avg_f1_nan))

    df = pd.DataFrame.from_dict(stat_metrics_per_split)
    df.loc['mean'] = df.mean().round(3)
    df.to_csv("{}/stats.csv".format(models_dir), index=True)

    confusion_mat = np.true_divide(confusion_mat, occurrence_mat)
    confusion_mat_per_group = np.true_divide(confusion_mat_per_group, occurrence_mat)

    make_confusion_matrices(confusion_mat, confusion_mat_per_group, df, folds, IGNORE_CLASSES, models_dir)
    # AfterMath(models_dir, classes, ignore_classes).run()


def make_confusion_matrices(confusion_mat, confusion_mat_per_group, df, folds, ignore_classes, models_dir):
    use_class_indices = [ci for ci, c in enumerate(classes) if c not in ignore_classes]
    final_classes = [c for ci, c in enumerate(classes) if c not in ignore_classes]
    save_confusion_mat(confusion_mat, df, final_classes, folds, models_dir, use_class_indices, 'component')
    save_confusion_mat(confusion_mat_per_group, df, final_classes, folds, models_dir, use_class_indices, 'group')


def save_confusion_mat(confusion_mat, df, final_classes, folds, models_dir, use_class_indices, item='component'):
    final_cm = confusion_mat[np.ix_(use_class_indices, use_class_indices)]
    final_mean_counts = np.zeros((final_cm.shape[0], 1))
    for idx, cat in enumerate(final_classes):
        final_mean_counts[idx, 0] = df.at['mean', 'Cnt{}Te{}'.format('c' if item == 'component' else 'g', cat)]
    show_confusion_matrix_with_counts(final_cm,
                                      final_mean_counts,
                                      final_classes,
                                      "Cross validation from {} splits.\n"
                                      "Sum(pct) per row = 1.\n"
                                      "Avg count across {} splits shown on right bar.".format(item, len(folds)),
                                      os.path.join(models_dir, "conf_mat_{}.jpg".format(item)),
                                      True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--models_dir", required=True, type=str)
    parser.add_argument("--svm_impl", default="SVM", type=str)
    parser.add_argument("--unique_dirs", type=str)
    parser.add_argument("--classes", type=str, default="")
    parser.add_argument("--ignore_classes", type=str, default="")
    parser.add_argument("--avg_f1_nan", type=str, default="np.nan", help="e.g. use np.nan or 0")
    args = parser.parse_args()

    if args.classes != "":  # note that classes must be aligned with _labels.npy so be careful in using this
        classes = args.classes.split(",")
    if args.ignore_classes != "":
        IGNORE_CLASSES = args.ignore_classes.split(",")

    folds = []
    for file_or_dir in os.listdir(args.data_dir):
        num = "".join([s for s in file_or_dir if s.isdigit()])
        if "train" in file_or_dir:
            folds.append(int(num))

    run(args.data_dir, args.models_dir, folds, args.svm_impl)
