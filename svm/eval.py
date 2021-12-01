import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from svm.visual import show_confusion_matrix_with_counts

pd.set_option('display.max_columns', None)


def precision_recall(test_y, test_p, categories):
    if not isinstance(test_y, np.ndarray):
        assert isinstance(test_y, list)
        test_y = np.array(test_y)
    if not isinstance(test_p, np.ndarray):
        assert isinstance(test_p, list)
        test_p = np.array(test_p)
    assert test_y.ndim == 1 and test_p.ndim == 1
    tp = {}
    fn = {}
    fp = {}
    metrics = {}
    for idx, cat in enumerate(categories):
        tp[cat] = np.sum(np.bitwise_and(test_y == idx, test_p == idx))
        fn[cat] = np.sum(np.bitwise_and(test_y == idx, test_p != idx))
        fp[cat] = np.sum(np.bitwise_and(test_y != idx, test_p == idx))
        if tp[cat] + fp[cat] > 0:
            precision = np.round(tp[cat] / (tp[cat] + fp[cat]), 3)
        else:
            precision = np.nan
        if tp[cat] + fn[cat] > 0:
            recall = np.round(tp[cat] / (tp[cat] + fn[cat]), 3)
        else:
            recall = np.nan
        metrics["Precision{}".format(cat)] = precision
        metrics["Recall{}".format(cat)] = recall
    return metrics


def get_normalized_confusion_matrix(test_y, test_p, classes):
    cm = confusion_matrix(test_y, test_p, labels=list(range(len(classes))))
    normalized_cm = np.zeros_like(cm).astype(float)
    for row in range(len(classes)):
        total_true = np.sum(cm[row, :])
        if total_true != 0:
            for col in range(len(classes)):
                normalized_cm[row, col] = cm[row, col] / total_true
    return normalized_cm


# t_y = np.array([1,2,3,3,2,1])
# t_p = np.array([1,2,1,2,3,1])
# print(precision_recall(t_y, t_p, [1,2,3]))


class AfterMath:

    COMMON_STRUCTURES = [
        "window",
        "roof",
        "wall",
        "door",
        "column",
    ]
    ALL_STRUCTURES = [
        "window",
        "minare",
        "tower",
        "arch_bay",
        "buttress",
        "roof",
        "wall",
        "door",
        "ornament",
        "dome",
        "column",
        "pilaster",
        "balcony",
        "railing",
        "shutters"
    ]

    def __init__(self, cross_val_results_dir, classes, ignore_classes=[]):
        self.folder = cross_val_results_dir
        self.classes = classes
        self.ignore_classes = ignore_classes

    def run(self):
        all_dfs = []
        for file in os.listdir(self.folder):
            if ".predictions.csv" in file:
                fold_id = "".join([c for c in file if c.isdigit()])
                df = pd.read_csv(os.path.join(self.folder, file))
                df['fold_id'] = fold_id
                all_dfs.append(df)
        df = pd.concat(all_dfs)
        self.stats_per_structure(df.copy())
        df.to_csv(os.path.join(self.folder, "all_predictions.csv"))

        self.confusion_matrix_per_building(df)
        self.accuracies_per_building_group(df)

    def stats_per_structure(self, df):
        stats = {}
        for structure in self.ALL_STRUCTURES:
            df[structure] = df.apply(lambda row: structure.lower() in row['name'].lower(), axis=1)
            df_structure = df[df[structure] == True]
            found = np.sum(df_structure['ground_truth'] == df_structure['prediction'])
            stats[structure] = [round(found/df_structure.shape[0], 3), df_structure.shape[0]]
        df['common'] = df[self.COMMON_STRUCTURES].any(axis=1)
        common_df = df[df['common'] == True]
        distinct_df = df[df['common'] == False]
        found = np.sum(common_df['ground_truth'] == common_df['prediction'])
        stats['common'] = [round(found/common_df.shape[0], 3), common_df.shape[0]]
        found = np.sum(distinct_df['ground_truth'] == distinct_df['prediction'])
        stats['distinct'] = [round(found/distinct_df.shape[0], 3), distinct_df.shape[0]]
        reindex = self.COMMON_STRUCTURES + ['common'] + \
                  [s for s in self.ALL_STRUCTURES if s not in self.COMMON_STRUCTURES] + ['distinct']
        pd.DataFrame\
            .from_dict(stats, orient='index', columns=['Accuracy', 'Occurrences'])\
            .reindex(reindex)\
            .to_csv(os.path.join(self.folder, "stats_per_structure.csv"))

    def confusion_matrix_per_building(self, df):
        for building, data in df.groupby(by=['building']):
            ground_truths = data['ground_truth']
            predictions = data['prediction']
            folds = len(np.unique(data['fold_id'].values))
            counts = np.zeros((len(self.classes), 1))
            for idx in range(len(self.classes)):
                counts[idx] = np.sum(ground_truths == idx)
            ncm = get_normalized_confusion_matrix(ground_truths, predictions, self.classes)

            use_class_indices = [ci for ci, c in enumerate(self.classes) if c not in self.ignore_classes]
            final_classes = [c for ci, c in enumerate(self.classes) if c not in self.ignore_classes]

            final_ncm = ncm[np.ix_(use_class_indices, use_class_indices)]
            final_counts = counts[np.ix_(use_class_indices)]

            show_confusion_matrix_with_counts(final_ncm,
                                              final_counts,
                                              final_classes,
                                              "{}\nAppears in {} splits.\n"
                                              "Total component counts shown in right.".format(building, folds),
                                              os.path.join(self.folder, "{}.jpg".format(building)),
                                              False)

    def accuracies_per_building_group(self, df):
        accuracy_per_building_group = {}
        for (building, group), group_df in df.groupby(by=['building', 'group']):
            cnt_correct = np.sum(group_df['ground_truth'] == group_df['prediction'])
            accuracy_per_building_group.setdefault(building, {})
            accuracy_per_building_group[building][group] = cnt_correct / group_df.shape[0]
        for building in accuracy_per_building_group:
            accuracy_per_building_group[building]['avg'] = np.nanmean(list(accuracy_per_building_group[building].values()))
        with open(os.path.join(self.folder, "building_group_accuracies.json"), "w") as fout:
            json.dump(accuracy_per_building_group, fout, indent=2)


def _per_building():
    overall = {}
    # pb = json.load(open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_svm/models/buildnet_mink/annfass_ply_100K/layer_n-3_features.backup/sum.max/classification_cross_val/building_group_accuracies.json", "r"))
    pb = json.load(open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_svm/models/buildnet_mink/annfass_ply_100K/layer_n-2_features/avg/classification_cross_val/building_group_accuracies.json", "r"))
    for key, value in pb.items():
        if 'avg' in value:
            overall[key] = value['avg']
    df = pd.DataFrame.from_dict(overall, orient='index')
    # df.to_csv("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_svm/models/buildnet_mink/annfass_ply_100K/layer_n-3_features.backup/sum.max/classification_cross_val/building_group_accuracies.csv", sep=';')
    df.to_csv("/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_svm/models/buildnet_mink/annfass_ply_100K/layer_n-2_features/avg/classification_cross_val/building_group_accuracies.csv", sep=';')


def f1score(eval_file, avg_f1_nan=0, ignore_classes='Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown'):
    df = pd.read_csv(eval_file, header=0, index_col=0)
    with open(eval_file.replace("eval.csv", "avg_f1_score.txt"), "w") as fout:
        mean_precisions = []
        mean_recalls = []
        mean_fc1scores = []
        for col in df.columns:
            if col.startswith("cP") and not any(ic in col for ic in ignore_classes.split(",")):
                mean_precisions.append(df.at['mean', col])
            elif col.startswith("cR") and not any(ic in col for ic in ignore_classes.split(",")):
                mean_recalls.append(df.at['mean', col])
        for p, r in zip(mean_precisions, mean_recalls):
            if np.isnan(p) or np.isnan(r):
                mean_fc1scores.append(avg_f1_nan)
            else:
                if p + r == 0:
                    mean_fc1scores.append(0)
                else:
                    mean_fc1scores.append(2 * p * r / (p + r))
        print("total average per component: {}\n".format(np.round(np.nanmean(mean_fc1scores), 3)))
        fout.write("total average per component: {}\n".format(np.round(np.nanmean(mean_fc1scores), 3)))

        mean_precisions = []
        mean_recalls = []
        mean_fc1scores = []
        for col in df.columns:
            if col.startswith("gP") and not any(ic in col for ic in ignore_classes.split(",")):
                mean_precisions.append(df.at['mean', col])
            elif col.startswith("gR") and not any(ic in col for ic in ignore_classes.split(",")):
                mean_recalls.append(df.at['mean', col])
        for p, r in zip(mean_precisions, mean_recalls):
            if np.isnan(p) or np.isnan(r):
                mean_fc1scores.append(avg_f1_nan)
            else:
                if p + r == 0:
                    mean_fc1scores.append(0)
                else:
                    mean_fc1scores.append(2 * p * r / (p + r))
        print("total average per group: {}\n".format(np.round(np.nanmean(mean_fc1scores), 3)))
        fout.write("total average per group: {}\n".format(np.round(np.nanmean(mean_fc1scores), 3)))


def over_multiple_experiments(root, model, svm_impl, layer=None):
    f1_per_component = []
    f1_per_unique_component = []
    for repeat_dir in os.listdir(root):
        if f"svm_from_{model}" in repeat_dir and os.path.isdir(os.path.join(root, repeat_dir)):
            if layer is not None:
                f1_file = os.path.join(root, repeat_dir, f"svm_{svm_impl}", layer, "avg_f1_score.txt")
            else:
                f1_file = os.path.join(root, repeat_dir, f"svm_{svm_impl}", "avg_f1_score.txt")
            with open(f1_file, "r") as fin:
                score = np.float(fin.readline().rstrip().split(": ")[1])
                f1_per_component.append(score)
                score = np.float(fin.readline().rstrip().split(": ")[1])
                f1_per_unique_component.append(score)
    print(f1_per_component)
    print(f1_per_unique_component)
    print(np.mean(f1_per_component), np.max(f1_per_component))
    print(np.mean(f1_per_unique_component), np.max(f1_per_unique_component))


def over_multiple_experiments_pr(root, model, svm_impl, layer=None,
                                 ignore='Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown',
                                 per_component=True):
    precisions = {}
    recalls = {}
    ignore = ignore.split(",")
    for repeat_dir in os.listdir(root):
        if f"svm_from_{model}" in repeat_dir and os.path.isdir(os.path.join(root, repeat_dir)):
            if layer is not None:
                eval_file = os.path.join(root, repeat_dir, f"svm_{svm_impl}", layer, "eval.csv")
            else:
                eval_file = os.path.join(root, repeat_dir, f"svm_{svm_impl}", "eval.csv")
            df = pd.read_csv(eval_file, header=0, index_col=0)
            for column in df.columns:
                if any(i.lower() in column.lower() for i in ignore):
                    continue
                if (per_component and column.startswith('cP')) or (not per_component and column.startswith('gP')):
                    precisions.setdefault(column, [])
                    precisions[column].append(df.at['mean', column])
                if (per_component and column.startswith('cR')) or (not per_component and column.startswith('gR')):
                    recalls.setdefault(column, [])
                    recalls[column].append(df.at['mean', column])
    for column in precisions:
        precisions[column] = np.round(np.mean(precisions[column]), 3)
    for column in recalls:
        recalls[column] = np.round(np.mean(recalls[column]), 3)
    print(precisions)
    print(recalls)
    out_file = os.path.join(root, f"svm_from_{model}_"+f"svm_{svm_impl}_"+(layer or "")+"result.txt")
    with open(out_file, "w") as fout:
        fout.write(str(precisions)+"\n"+str(recalls)+"\n")


if __name__ == '__main__':
    # _per_building()
    #
    # model_name = "ae2D_bn_on_buildnet"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_unique")):
    #         eval_file = os.path.join(input_dir, "svm_unique", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "ae2D_bn_on_buildnet", "unique", "z_dim_avg_as_is")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "ae2D_bn_on_buildnet", "unique", "z_dim_all_as_is", per_component=False)
    # repeat 1 is the best. avg f1 score is 0.206, max is 0.231
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "ae2D_bn_on_buildnet", "random_unique", "z_dim_avg_as_is")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "ae2D_bn_on_buildnet", "random_unique", "z_dim_all_as_is", per_component=False)
    # repeat 3 is the best. avg f1 score is 0.082, max is 0.101

    # model_name = "3daae_on_buildnet"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_simple")):
    #         eval_file = os.path.join(input_dir, "svm_simple", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           model_name, "simple", "z_dim_all_as_is")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              model_name, "simple", "z_dim_all_as_is")
    # repeat 1 is the best. avg f1 score is 0.085, max is 0.098
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "3daae_on_buildnet", "random", "z_dim_all_as_is")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "3daae_on_buildnet", "random", "z_dim_all_as_is")
    # repeat 3 is the best. avg f1 score is 0.824

    # model_name = "decor_original"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_simple")):
    #         eval_file = os.path.join(input_dir, "svm_simple", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           model_name, "simple", "discr_all_max" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              model_name, "simple", "discr_all_max" )
    # repeat 2 is the best. avg f1 score is 0.101, max is 0.106
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "decor_original", "random", "discr_all_max" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "decor_original", "random", "discr_all_max" )
    # repeat 2 is the best. avg f1 score is 0.082

    # model_name = "ocnn_depth6_nocolour_moredata"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_simple")):
    #         eval_file = os.path.join(input_dir, "svm_simple", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           model_name, "simple", "feature_concat_as_is_per_component_max" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              model_name, "simple", "feature_concat_as_is_per_component_max")
    # repeat 0 is the best. avg f1 score is 0.107, max is 0.12
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "ocnn_depth6_nocolour_moredata", "random", "feature_concat_as_is_per_component_max" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "ocnn_depth6_nocolour_moredata", "random", "feature_concat_as_is_per_component_max")
    # repeat 0 is the best. avg f1 score is 0.075

    # model_name = "mink_vae_on_buildnet_component_balanced"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_simple")):
    #         eval_file = os.path.join(input_dir, "svm_simple", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           model_name, "simple", "z_dim_all_as_is" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              model_name, "simple", "z_dim_all_as_is")
    # repeat 1 is the best. avg f1 score is 0.139, max is 0.147
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "mink_vae_on_buildnet_component_balanced", "random", "z_dim_all_as_is" )
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "mink_vae_on_buildnet_component_balanced", "random", "z_dim_all_as_is")
    # repeat 3 is the best. avg f1 score is 0.078
    #
    # model_name = "mink_marios_again"
    # for repeat in range(5):
    #     init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
    #     input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
    #     for sub_dir in os.listdir(os.path.join(input_dir, "svm_simple")):
    #         eval_file = os.path.join(input_dir, "svm_simple", sub_dir, "eval.csv")
    #         if os.path.isfile(eval_file) and os.path.exists(eval_file):
    #             f1score(eval_file)
    #
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           model_name, "simple", "layer_n-2_features_max_per_component_avg")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              model_name, "simple", "layer_n-2_features_max_per_component_avg")
    # repeat 0 is the best. avg f1 score is 0.115, max is 0.128
    # over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                           "mink_marios", "random", "layer_n-2_features_max_per_component_avg")
    # over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
    #                              "mink_marios", "random", "layer_n-2_features_max_per_component_avg")
    # repeat 1 is the best. avg f1 score is 0.080

    model_name = "hog"
    for repeat in range(5):
        init_dir = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data"
        input_dir = f"{init_dir}/repeat_{repeat}_svm_from_{model_name}"
        for sub_dir in os.listdir(os.path.join(input_dir, "svm_hog_unique")):
            eval_file = os.path.join(input_dir, "svm_hog_unique", sub_dir, "eval.csv")
            if os.path.isfile(eval_file) and os.path.exists(eval_file):
                f1score(eval_file)

    over_multiple_experiments("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
                              model_name, "hog_unique", "as_is")
    over_multiple_experiments_pr("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data",
                                 model_name, "hog_unique", "as_is", per_component=False)
    # repeat 1 is the best. avg f1 score is 0.225, max is 0.236

    # f1score("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data/repeat_3_svm_from_ae2D_bn_on_buildnet/svm_unique/z_dim_all_as_is/eval.csv",
    #         avg_f1_nan=np.nan)
    # f1score("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/labeled_data/repeat_3_svm_from_ae2D_bn_on_buildnet/svm_unique/z_dim_all_as_is/eval.csv",
    #         avg_f1_nan=0)
