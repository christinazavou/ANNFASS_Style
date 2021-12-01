import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.pair_loader import get_loader


def train_split(train_data_loader, test_data_loader, exp_dir):
    pkl_filename = os.path.join(exp_dir, f"model.pkl")
    test_accuracies = train_test(train_data_loader, test_data_loader, pkl_filename)
    return test_accuracies


def train_test(train_data_loader, test_data_loader, pkl_filename):
    train_anchors, train_queries, train_labels = next(iter(train_data_loader))
    test_anchors, test_queries, test_labels = next(iter(test_data_loader))

    train_anchors, train_queries, train_labels = np.array(train_anchors), np.array(train_queries), np.array(train_labels)
    test_anchors, test_queries, test_labels = np.array(test_anchors), np.array(test_queries), np.array(test_labels)

    train_x = np.hstack([train_anchors, train_queries])
    train_y = train_labels

    test_x = np.hstack([test_anchors, test_queries])
    test_y = test_labels

    if os.path.exists(pkl_filename):
        with open(pkl_filename, 'rb') as file:
            clf = pickle.load(file)
            print("Loaded {}".format(pkl_filename))
    else:
        clf = SVC(class_weight='balanced')
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(train_x, train_y)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)
        print("Trained and saved balanced SVC in {}".format(pkl_filename))

    test_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, test_pred)
    return acc


def main():
    test_accuracies = {}
    test_lengths = {}
    for split in range(args.splits):
        triplet_train_path = os.path.join(args.splits_dir, f"train_triplets_{split}.txt")
        triplet_test_path = os.path.join(args.splits_dir, f"test_triplets_{split}.txt")
        train_data_loader, test_data_loader = get_loader(args, triplet_train_path, triplet_test_path)
        exp_dir = os.path.join(args.result_dir, args.exp_name, f"{split}")
        os.makedirs(exp_dir, exist_ok=True)
        accuracies = train_split(train_data_loader, test_data_loader, exp_dir)
        test_accuracies[split] = accuracies
        test_lengths[split] = next(iter(test_data_loader))[0].shape[0]
    overall_accuracy = 0
    for split in test_accuracies:
        overall_accuracy += test_accuracies[split]
    overall_accuracy /= len(test_accuracies)
    overall_accuracy = np.round(overall_accuracy, 3)
    print(f"overall_result: {overall_accuracy}")
    exp_fout = os.path.join(args.result_dir, args.exp_name, "overall.txt")
    with open(exp_fout, "w") as fout:
        fout.write(f"{overall_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--splits_dir', type=str, )
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--encodings_path', type=str, )
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')

    args = parser.parse_args()

    main()
