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


class SVM:

    def __init__(self, train_csv, test_csv, classes, pkl_filename, ignore_classes=None,):
        if train_csv:
            self.train_loader = iter(StylisticImagesEncodingsDataLoader(train_csv, batch_size=5000, num_workers=0))
        self.test_loader = iter(StylisticImagesEncodingsDataLoader(test_csv, batch_size=5000, num_workers=0))
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

        class_indices = [idx for idx, classname in enumerate(self.classes) if list(train_y).count(idx) > 0
                         and classname not in self.ignore_classes]
        classes = [classname for idx, classname in enumerate(self.classes) if idx in class_indices]
        print("classes to be used: {}".format(classes))

        train_indices = [i for i, y in enumerate(train_y) if y in class_indices]
        test_indices = [i for i, y in enumerate(test_y) if y in class_indices]

        # note: train_names contains the full component name e.g. 01_Cathedral/group1_Window_Colonial_Mesh
        train_x, train_y, train_names = train_x[train_indices, :], train_y[train_indices], train_names[train_indices]
        test_x, test_y, test_names = test_x[test_indices, :], test_y[test_indices], test_names[test_indices]

        return (train_x, train_y, test_x, test_y, test_names), class_indices

    def load(self):
        if os.path.exists(self.pkl_filename):
            with open(self.pkl_filename, 'rb') as file:
                clf = pickle.load(file)
                print("Loaded {}".format(self.pkl_filename))
                return clf
        return False

    def train(self, train_x, train_y):
        clf = self.load()
        if not clf:
            # note: svm.SVC accepts any categorical input for y .. & unique classes can be accessed by clf.classes_
            clf = svm.SVC(class_weight=self.class_weight)
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

    def evaluate_wrapper(self, test_y, test_p):
        metrics = {}
        results, cmn = self.evaluate(test_y, test_p)
        for cat in self.classes:
            metrics['cP{}'.format(cat)] = results['Precision{}'.format(cat)]
            metrics['cR{}'.format(cat)] = results['Recall{}'.format(cat)]
        return metrics, cmn

    def save_predictions(self, test_y, test_p, names):
        data = {'name': [], 'ground_truth': [], 'prediction': [], 'model': []}

        def add_values(d: dict, comp_name: str, gt: int, pred: int):
            d['name'].append(comp_name)
            d['ground_truth'].append(gt)
            d['prediction'].append(pred)
            d['model'].append(comp_name)

        for y, p, name in zip(test_y, test_p, names):
            add_values(data, name, y, p)

        all_df = pd.DataFrame().from_dict(data)
        all_df.to_csv(self.predictions_fname, index=False)

    def run(self):
        metrics = {}

        (train_x, train_y, test_x, test_y, test_n), class_indices = self.load_data()

        metrics.update(calculate_class_percentages(self.classes, train_y, 'cTr'))
        metrics.update(calculate_class_percentages(self.classes, test_y, 'cTe'))

        if len(np.unique(train_y)) < 2:
            raise Exception("Can't train an SVM given less than 2 labels in training data")
        if len(np.unique(train_y)) > len(np.unique(test_y)):
            print("Warning: not all training classes appear in test data")

        clf = self.train(train_x, train_y)
        test_p = clf.predict(test_x)
        self.save_predictions(test_y, test_p, test_n)

        results, cmn = self.evaluate_wrapper(test_y, test_p)
        metrics.update(results)

        class_occurs_mat = self.get_class_occupancy_matrix(cmn, test_y)

        return metrics, cmn, class_occurs_mat

    def get_class_occupancy_matrix(self, cmn, test_y):
        class_occurs_mat = np.zeros_like(cmn).astype(int)
        for class_1 in range(len(self.classes)):
            if class_1 in test_y:
                for class_2 in range(len(self.classes)):
                    if class_2 in test_y:
                        class_occurs_mat[class_1, class_2] = 1
        return class_occurs_mat

    def load_inference_data(self):
        test_x, test_y, test_names = next(self.test_loader)
        print("test data loaded: x{}".format(test_x.shape))

        return test_x, test_y, test_names

    def infer(self):
        test_x, _, test_n = self.load_inference_data()
        clf = self.load()
        if not clf:
            raise Exception("Couldnt load pkl ", self.pkl_filename)
        test_p = clf.predict(test_x)
        result = {n: p for n, p in zip(test_n, test_p)}
        return result
