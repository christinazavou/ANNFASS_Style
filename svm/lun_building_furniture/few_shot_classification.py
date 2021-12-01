import argparse
import os
import random
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from sklearn_impl.eval import precision_recall, get_normalized_confusion_matrix
from sklearn_impl.visual import show_confusion_matrix


balanced_accuracy = 0.


def run(data_dir, models_dir, ):

    models = {}
    models_per_style = {}
    for model_dir in os.listdir(data_dir):
        if model_dir in style_per_building:
            style = style_per_building[model_dir]
            models_per_style.setdefault(style, [])
            models_per_style[style] += [model_dir]
            models[model_dir] = os.path.join(data_dir, model_dir, "whole.npy")

    few_shot_scenarios = {}
    for few_shot in [1, 5, 10]:
    # for few_shot in [1, 2, 3]:
        few_shot_scenarios.setdefault(few_shot, {})
        for scenario_id in range(100):
            few_shot_scenarios[few_shot].setdefault(scenario_id, {})
            example_models = {}
            for style in models_per_style:
                random.shuffle(models_per_style[style])
                example_models[style] = models_per_style[style][:few_shot]
            test_models = [m for m in models if not any(m in examples for examples in example_models.values())]
            assert len(test_models) == len(models) - len(classes) * few_shot
            few_shot_scenarios[few_shot][scenario_id] = {'examples': example_models, 'test': test_models}

    for few_shot, scenarios in few_shot_scenarios.items():

        confusion_mat = np.zeros((len(classes), len(classes)))
        recall_precisions = {}
        for style in classes:
            recall_precisions[f'Recall{style}'] = []
            recall_precisions[f'Precision{style}'] = []

        for scenario_id, scenario in scenarios.items():
            class_representations = {}
            for style in classes:
                class_representations.setdefault(style, [])
                for example_model in scenario['examples'][style]:
                    filename = models[example_model]
                    class_representations[style] += [np.load(filename)]
                class_representations[style] = np.array(class_representations[style])
                class_representations[style] = np.mean(class_representations[style], 0)

            class_vectors = np.array(list(class_representations.values()))

            predictions = []
            targets = []
            for test_model in scenario['test']:
                test_vector = np.load(models[test_model])
                distances = cdist(class_vectors, test_vector.reshape((1, -1)))
                predictions.append(np.argmax(distances))
                targets.append(classes.index(style_per_building[test_model]))

            metrics = precision_recall(targets, predictions, classes)
            for metric in metrics:
                if metric in recall_precisions:
                    recall_precisions[metric].append(metrics[metric])
            cmn = get_normalized_confusion_matrix(targets, predictions, classes)
            confusion_mat += cmn

        os.makedirs(f"{models_dir}", exist_ok=True)
        print(f"{few_shot}-shot: \n")
        with open(f"{models_dir}/{few_shot}-shot-eval.txt", "w") as fout:
            f1scores = []
            for style in classes:
                recall = np.round(np.nanmean(recall_precisions[f'Recall{style}']), 3)
                precision = np.round(np.nanmean(recall_precisions[f'Precision{style}']), 3)
                print(f"Recall{style}:\t{recall}")
                fout.write(f"Recall{style}:\t{recall}\n")
                print(f"Precision{style}:\t{precision}")
                fout.write(f"Precision{style}:\t{precision}\n")
                f1score = np.round(2 * recall * precision / (recall + precision), 3)
                f1scores.append(f1score)
                print(f"F1score{style}:\t{f1score}")
                fout.write(f"F1score{style}:\t{f1score}\n")
            avgf1score = np.nanmean(f1scores)
            print(f"average F1score:\t{avgf1score}")
            fout.write(f"average F1score:\t{avgf1score}")
        confusion_mat = np.true_divide(confusion_mat, 100)
        show_confusion_matrix(confusion_mat, classes, "", f"{models_dir}/{few_shot}-shot-cmat.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--models_dir", required=True, type=str)
    parser.add_argument("--classes", required=True, type=str, help="e.g. C1,C2,C3,C4,C5,C6,C7,C8")
    parser.add_argument("--label_file",
                        default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/building/labels.txt",
                        type=str)
    args = parser.parse_args()

    classes = args.classes.split(",")

    label_file = args.label_file
    style_per_building = {}
    with open(label_file, "r") as fin:
        for line in fin.readlines():
            building_file, style = line.rstrip().split(" ")
            if int(style) < len(classes):
                style_per_building[building_file.replace(".ply", "")] = classes[int(style)]

    folds = []
    run(args.data_dir, args.models_dir)
