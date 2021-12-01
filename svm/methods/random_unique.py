import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import svm

from svm.eval import precision_recall, get_normalized_confusion_matrix
from svm.utils.datasets import StylisticImagesEncodingsDataLoader


def calculate_class_percentages(classes, train_ys, prefix='Tr'):
    metrics = {}
    for label, class_name in enumerate(classes):
        cnt = int(list(train_ys).count(label))
        metrics['Cnt{}{}'.format(prefix, class_name)] = cnt
        metrics['Pct{}{}'.format(prefix, class_name)] = round(cnt / len(train_ys), 3)
    return metrics


def get_sample_weights(classes, labels_pct, test_ys):
    sample_weights = []
    for label in test_ys:
        if labels_pct[classes[label]] == 0:
            sample_weights.append(0)
        else:
            sample_weights.append(1/labels_pct[classes[label]])
    return sample_weights


class RandomPredictor:

    def __init__(self, train_csv, test_csv, classes, pkl_filename, ignore_classes=None, unique_dirs=None):
        self.train_loader = iter(StylisticImagesEncodingsDataLoader(train_csv, batch_size=5000, num_workers=0))
        self.test_loader = iter(StylisticImagesEncodingsDataLoader(test_csv, batch_size=5000, num_workers=0))
        self.classes = classes
        self.pkl_filename = pkl_filename
        self.predictions_fname = pkl_filename.replace(".pkl", ".predictions.csv")
        self.ppb_fname = pkl_filename.replace(".pkl", ".ppbuildings.json")
        self.ignore_classes = ignore_classes
        self.class_weight = 'balanced'
        self.unique_dirs = unique_dirs

    def load_data(self):
        train_x, train_y, train_names = next(self.train_loader)
        test_x, test_y, test_names = next(self.test_loader)
        print("train data loaded: x {}, y: {} test data loaded: x{}, y: {}".format(
            train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        class_indices = [idx for idx, classname in enumerate(self.classes) if list(train_y).count(idx) > 0
                         and classname not in self.ignore_classes]
        classes = [classname for idx, classname in enumerate(self.classes) if idx in class_indices]
        print("classes to be used: {}".format(classes))

        train_indices = [i for i, y in enumerate(train_y) if y in class_indices]
        test_indices = [i for i, y in enumerate(test_y) if y in class_indices]

        # note: train_names contains the full component name e.g. 01_Cathedral/group1_Window_Colonial_Mesh
        train_x, train_y, train_names = train_x[train_indices, :], train_y[train_indices], train_names[train_indices]
        test_x, test_y, test_names = test_x[test_indices, :], test_y[test_indices], test_names[test_indices]

        train_grouped = self.group_data_on_unique_component(train_names, train_x, train_y, self.unique_dirs)
        test_grouped = self.group_data_on_unique_component(test_names, test_x, test_y, self.unique_dirs)

        return (train_x, train_y, test_x, test_y, test_names), (train_grouped, test_grouped), class_indices

    @staticmethod
    def group_data_on_unique_component(names, x, y, unique_dirs):
        # names_tuples = [(i, n.split('/')[0], "_".join(n.split('/')[1].split('_')[1:])) for i, n in enumerate(names)]
        names_tuples = [(i, n.split('/')[0], n.split('/')[1]) for i, n in enumerate(names)]
        initial_cnt = len(names)
        final_cnt = 0

        found_indices = set()

        grouped = []
        for unique_dir in unique_dirs:
            for building in os.listdir(unique_dir):
                building_names_tuples = [t for t in names_tuples if t[1].replace("_refinedTextures","") == building.replace("_refinedTextures", "")]
                unique_file = os.path.join(unique_dir, building, "duplicates.json")
                if building_names_tuples and os.path.exists(unique_file):
                    building_unique_dict = json.load(open(unique_file, "rb"))
                    for unique_component_id, duplicate_components in building_unique_dict.items():
                        # duplicate_components = [dc.split(".")[0] for dc in duplicate_components]
                        duplicate_components = [dc.split(".")[0].replace("style_mesh_", "") for dc in duplicate_components]

                        current_tuples = [t for t in building_names_tuples if any(dc in t[2].split(".")[0] for dc in duplicate_components)]
                        current_indices = [t[0] for t in current_tuples]
                        if set(current_indices) & found_indices != set():
                            if (len(current_indices)) == len(set(current_indices) & found_indices):
                                continue
                            raise Exception("Some of {} appears in more than one group".format(names[current_indices]))
                        found_indices.update(set(current_indices))
                        if len(current_indices):
                            final_cnt += len(current_indices)
                            grouped.append((x[current_indices, :],
                                            y[current_indices],
                                            names[current_indices],
                                            current_indices,
                                            "{}_{}".format(building, unique_component_id)))
        # if initial_cnt != final_cnt:
        #     not_found_indices = list(set(list(range(len(names)))) - found_indices)
        #     raise Exception("not found: {}".format(names[not_found_indices]))

        return grouped

    def get_normalized_confusion_matrix(self, test_y, test_p):
        return get_normalized_confusion_matrix(test_y, test_p, self.classes)

    def evaluate(self, test_y, test_p):
        metrics = precision_recall(test_y, test_p, self.classes)
        normalized_cm = self.get_normalized_confusion_matrix(test_y, test_p)
        return metrics, normalized_cm

    def evaluate_wrapper(self, test_y, test_p, per_group_test_y, per_group_test_p):
        metrics = {}
        results, cmn = self.evaluate(test_y, test_p)
        for cat in self.classes:
            metrics['cP{}'.format(cat)] = results['Precision{}'.format(cat)]
            metrics['cR{}'.format(cat)] = results['Recall{}'.format(cat)]
        per_group_results, per_group_cmn = self.evaluate(per_group_test_y, per_group_test_p)
        for cat in self.classes:
            metrics['gP{}'.format(cat)] = per_group_results['Precision{}'.format(cat)]
            metrics['gR{}'.format(cat)] = per_group_results['Recall{}'.format(cat)]
        return metrics, cmn, per_group_cmn

    def save_predictions(self, test_y, test_p, names):
        data = {'name': [], 'ground_truth': [], 'prediction': [], 'building': [], 'group': []}

        def add_values(d: dict, comp_name: str, gt: int, pred: int):
            d['name'].append(comp_name)
            d['ground_truth'].append(gt)
            d['prediction'].append(pred)
            d['building'].append(comp_name.split("/")[0])
            d['group'].append(comp_name.split("/")[1].split("_")[0])

        for y, p, name in zip(test_y, test_p, names):
            add_values(data, name, y, p)

        all_df = pd.DataFrame().from_dict(data)
        all_df.to_csv(self.predictions_fname, index=False)

    def predict(self, test_x, train_y):
        classes = set(train_y)
        test_p = np.random.choice(np.array(list(classes)), len(test_x))
        return test_p

    def run(self):
        metrics = {}

        (train_x, train_y, test_x, test_y, test_n), (train_grouped, test_grouped), class_indices = self.load_data()

        metrics.update(calculate_class_percentages(self.classes, train_y, 'cTr'))
        metrics.update(calculate_class_percentages(self.classes, test_y, 'cTe'))

        if len(np.unique(train_y)) < 2:
            raise Exception("Can't train an SVM given less than 2 labels in training data")
        if len(np.unique(train_y)) > len(np.unique(test_y)):
            print("Warning: not all training classes appear in test data")

        test_p = self.predict(test_x, train_y)
        self.save_predictions(test_y, test_p, test_n)

        per_group_test_p, per_group_test_y = self.get_test_yp_per_group(test_grouped, test_p)
        per_group_train_y = self.get_train_y_per_group(train_grouped)
        metrics.update(calculate_class_percentages(self.classes, per_group_test_y, prefix='gTe'))
        metrics.update(calculate_class_percentages(self.classes, per_group_train_y, prefix='gTr'))

        results, cmn, per_group_cmn = self.evaluate_wrapper(test_y, test_p, per_group_test_y, per_group_test_p)
        metrics.update(results)

        class_occurs_mat = self.get_class_occupancy_matrix(cmn, test_y)
        class_occurs_mat_per_group = self.get_class_occupancy_matrix(cmn, per_group_test_y)
        if not np.equal(class_occurs_mat, class_occurs_mat_per_group).all():
            print("Warning: not equal class occurrence per component vs per group(unique component)")

        return metrics, cmn, per_group_cmn, class_occurs_mat

    def get_class_occupancy_matrix(self, cmn, test_y):
        class_occurs_mat = np.zeros_like(cmn).astype(int)
        for class_1 in range(len(self.classes)):
            if class_1 in test_y:
                for class_2 in range(len(self.classes)):
                    if class_2 in test_y:
                        class_occurs_mat[class_1, class_2] = 1
        return class_occurs_mat

    @staticmethod
    def get_test_yp_per_group(test_grouped, test_p):
        per_group_test_y = []
        per_group_test_p = []
        for (xs, ys, names, indices, unique_group) in test_grouped:
            assert len(np.unique(ys)) == 1, "group {} contains multiple classes {}".format(unique_group, ys)
            values, counts = np.unique(test_p[indices], return_counts=True)
            ind = np.random.choice(np.flatnonzero(counts == counts.max()))  # tie breaking by random choice
            # i.e. if test_p[i] = [2,3,3,2] then we have values = [2,3] and counts = [2,2]
            # therefore we can pick 2 as most common or 3 as most common!
            # instead, if we would use ind = np.argmax(counts) it would always pick first choice i.e. 2 as most common
            per_group_test_p.append(values[ind])
            per_group_test_y.append(ys[0])
        return per_group_test_p, per_group_test_y

    @staticmethod
    def get_train_y_per_group(train_grouped):
        per_group_train_y = []
        for (x, y, n, i, g) in train_grouped:
            assert len(np.unique(y)) == 1, "group {} contains multiple classes {}".format(g, y)
            per_group_train_y.append(y[0])
        return per_group_train_y
