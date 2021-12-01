import os
import pickle

import numpy as np
import pandas as pd
from sklearn import svm

from svm.eval import precision_recall, get_normalized_confusion_matrix
from svm.utils.datasets import StylisticComponentPlyWithCurvaturesDataLoader


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


class SVM:

    def __init__(self, train_csv, test_csv, classes, pkl_filename, ignore_classes=None):
        self.train_loader = iter(StylisticComponentPlyWithCurvaturesDataLoader(train_csv, batch_size=-1))
        self.test_loader = iter(StylisticComponentPlyWithCurvaturesDataLoader(test_csv, batch_size=-1))
        self.classes = classes
        self.pkl_filename = pkl_filename
        self.predictions_fname = pkl_filename.replace(".pkl", ".predictions.csv")
        self.ppb_fname = pkl_filename.replace(".pkl", ".ppbuildings.json")
        self.ignore_classes = ignore_classes
        self.class_weight = 'balanced'

    def load_data(self):
        train_x, train_y, train_names = next(self.train_loader)
        test_x, test_y, test_names = next(self.test_loader)
        print("train data loaded: x {}, y: {} test data loaded: x{}, y: {}".format(
            train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        unique_train_groups = np.unique(train_names)
        unique_test_groups = np.unique(test_names)
        assert set(unique_train_groups) & set(unique_test_groups) == set()
        print("Unique groups loaded: train: {}, test: {}".format(len(unique_train_groups), len(unique_test_groups)))

        class_indices = [idx for idx, classname in enumerate(self.classes) if list(train_y).count(idx) > 0
                         and classname not in self.ignore_classes]
        classes = [classname for idx, classname in enumerate(self.classes) if idx in class_indices]
        print("classes to be used: {}".format(classes))

        train_indices = [i for i, y in enumerate(train_y) if y in class_indices]
        test_indices = [i for i, y in enumerate(test_y) if y in class_indices]

        # note: train_names contains the full component name e.g. 01_Cathedral/group1_Window_Colonial_Mesh
        train_x, train_y, train_names = train_x[train_indices, :], train_y[train_indices], train_names[train_indices]
        test_x, test_y, test_names = test_x[test_indices, :], test_y[test_indices], test_names[test_indices]

        test_grouped = self.group_data(test_names, test_x, test_y, unique_test_groups)
        train_grouped = self.group_data(train_names, train_x, train_y, unique_train_groups)

        return (train_x, train_y, test_x, test_y), (train_grouped, test_grouped), class_indices

    @staticmethod
    def group_data(names, x, y, unique_groups):
        grouped = []
        for group in unique_groups:
            group_indices = []
            for idx, name in enumerate(names):
                if group == name:
                    group_indices.append(idx)
            if len(group_indices) > 0:
                assert len(np.unique(y[group_indices])) == 1
                grouped.append((x[group_indices, :], y[group_indices],
                                names[group_indices],
                                group_indices,
                                group))
        return grouped

    def train(self, train_x, train_y):
        if os.path.exists(self.pkl_filename):
            with open(self.pkl_filename, 'rb') as file:
                clf = pickle.load(file)
                print("Loaded {}".format(self.pkl_filename))
                return clf
        else:
            # note: svm.SVC accepts any categorical input for y .. & unique classes can be accessed by clf.classes_
            clf = svm.SVC(class_weight=self.class_weight, probability=True)
            clf.fit(train_x, train_y)
            with open(self.pkl_filename, 'wb') as file:
                pickle.dump(clf, file)
            print("Trained and saved balanced SVC in {}".format(self.pkl_filename))
            return clf

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
            d['group'].append(comp_name.split("/")[1])

        for y, p, name in zip(test_y, test_p, names):
            add_values(data, name, y, p)

        all_df = pd.DataFrame().from_dict(data)
        all_df.to_csv(self.predictions_fname, index=False)

    def run(self):
        metrics = {}

        (train_x, train_y, test_x, test_y), (train_grouped, test_grouped), class_indices = self.load_data()

        metrics.update(calculate_class_percentages(self.classes, train_y, 'cTr'))
        metrics.update(calculate_class_percentages(self.classes, test_y, 'cTe'))

        # test_x and test_y do not appear in the same order as in test_grouped thus they are not aligned with test_n
        test_x = np.vstack([x for (x, y, n, i, g) in test_grouped])
        test_y = np.hstack([y for (x, y, n, i, g) in test_grouped])
        test_n = np.hstack([n for (x, y, n, i, g) in test_grouped])

        if len(np.unique(train_y)) < 2:
            raise Exception("Can't train an SVM given less than 2 labels in training data")
        if len(np.unique(train_y)) > len(np.unique(test_y)):
            print("Warning: not all training classes appear in test data")

        clf = self.train(train_x, train_y)
        test_prob = clf.predict_proba(test_x)
        test_p = np.argmax(test_prob, 1)
        test_p = [class_indices[p] for p in test_p]
        self.save_predictions(test_y, test_p, test_n)

        per_group_test_p, per_group_test_y = self.get_test_yp_per_group(test_grouped, test_prob, class_indices)
        per_group_train_y = self.get_train_y_per_group(train_grouped)
        metrics.update(calculate_class_percentages(self.classes, per_group_test_y, prefix='gTe'))
        metrics.update(calculate_class_percentages(self.classes, per_group_train_y, prefix='gTr'))

        results, cmn, per_group_cmn = self.evaluate_wrapper(test_y, test_p, per_group_test_y, per_group_test_p)
        metrics.update(results)

        class_occurs_mat = self.get_class_occupancy_matrix(cmn, test_y)
        class_occurs_mat_per_group = self.get_class_occupancy_matrix(cmn, per_group_test_y)
        assert np.equal(class_occurs_mat, class_occurs_mat_per_group).all()

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
    def get_test_yp_per_group(test_grouped, test_prob, class_indices):
        per_group_test_y = []
        per_group_test_p = []
        for (x, y, n, i, g) in test_grouped:
            assert len(np.unique(y)) == 1, "group {} contains multiple classes {}".format(g, y)
            per_group_test_y.append(y[0])
            group_test_prob = np.mean(test_prob[i], 0)
            group_test_p = np.argmax(group_test_prob)
            group_test_p = class_indices[group_test_p]
            per_group_test_p.append(group_test_p)
        return per_group_test_p, per_group_test_y

    @staticmethod
    def get_train_y_per_group(train_grouped):
        per_group_train_y = []
        for (x, y, n, i, g) in train_grouped:
            assert len(np.unique(y)) == 1, "group {} contains multiple classes {}".format(g, y)
            per_group_train_y.append(y[0])
        return per_group_train_y
