import argparse
import os

import numpy as np

from utils.triplet_loader import get_loader


def unsupervised_split(train_data_loader, test_data_loader):
    train_anchors, train_pos, train_neg = next(iter(train_data_loader))
    test_anchors, test_pos, test_neg = next(iter(test_data_loader))

    train_anchors, train_pos, train_neg = np.array(train_anchors), np.array(train_pos), np.array(train_neg)
    test_anchors, test_pos, test_pos = np.array(test_anchors), np.array(test_pos), np.array(test_neg)

    anchors = np.vstack([train_anchors, test_anchors])
    pos = np.vstack([train_pos, test_pos])
    neg = np.vstack([train_neg, test_neg])

    sim_pos = np.einsum('ij,ij->i', anchors, pos) / (np.linalg.norm(anchors, axis=1) * np.linalg.norm(pos, axis=1))
    sim_neg = np.einsum('ij,ij->i', anchors, neg) / (np.linalg.norm(anchors, axis=1) * np.linalg.norm(neg, axis=1))

    good_predictions = sim_pos > sim_neg
    accuracy = np.sum(good_predictions).astype(np.float)

    print('Test Accuracy: {}'.format(accuracy / len(good_predictions)))
    print("****************")

    return accuracy / len(good_predictions)


def main():
    test_accuracies = {}
    test_lengths = {}
    for split in range(args.splits):
        triplet_train_path = os.path.join(args.splits_dir, f"train_triplets_{split}.txt")
        triplet_test_path = os.path.join(args.splits_dir, f"test_triplets_{split}.txt")
        train_data_loader, test_data_loader = get_loader(args, triplet_train_path, triplet_test_path)
        accuracy = unsupervised_split(train_data_loader, test_data_loader)
        test_accuracies[split] = accuracy
        test_lengths[split] = next(iter(test_data_loader))[0].shape[0]

    overall_accuracy = 0
    for split in test_accuracies:
        overall_accuracy += test_accuracies[split]
    overall_accuracy /= len(test_accuracies)
    overall_accuracy = np.round(overall_accuracy, 3)

    print(f"overall_accuracy: {overall_accuracy}")
    exp_fout = os.path.join(args.result_dir, "overall.txt")
    os.makedirs(args.result_dir, exist_ok=True)
    with open(exp_fout, "w") as fout:
        fout.write(f"{overall_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', type=str, )
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--encodings_path', type=str, )
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--batch_size', default=50000, type=int, )

    args = parser.parse_args()

    main()
