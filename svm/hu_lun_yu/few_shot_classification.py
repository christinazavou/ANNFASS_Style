import argparse
import os
import pickle
import random
import sys

import numpy as np
from scipy import spatial
from sklearn import svm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from sklearn_impl.eval import precision_recall, get_normalized_confusion_matrix
from sklearn_impl.visual import show_confusion_matrix


def get_data():
    models = {}
    models_per_style = {}
    style_per_model = {}
    model_names = os.listdir(models_dir)
    assert len(model_names) == expected, f"Missing files ? {len(model_names)} != {expected}"
    for model_name in model_names:
        style = [s for s in classes if s.lower() in model_name.lower()]
        if len(style) == 0:
            if model_name not in exclude:
                raise Exception(f"{model_name} has no style")
            continue
        style = style[0]
        models_per_style.setdefault(style, [])
        models_per_style[style] += [model_name]
        if "u_shape" in model_name:
            models[model_name] = os.path.join(encodings_dir, model_name.replace("u_shape", "u_shape_"), "whole.npy")
            assert os.path.exists(os.path.join(encodings_dir, model_name.replace("u_shape", "u_shape_"), "whole.npy")), f"missing {encodings_dir}/{model_name.replace('u_shape', 'u_shape_')}"
        else:
            assert os.path.exists(os.path.join(encodings_dir, model_name, "whole.npy")), f"missing {encodings_dir}/{model_name}"
            models[model_name] = os.path.join(encodings_dir, model_name, "whole.npy")
        style_per_model[model_name] = style
    return models_per_style, style_per_model, models


def get_few_shot_scenarios(models_per_style, models):
    few_shot_scenarios = {}
    # for few_shot in [5, 10, 20]:
    for few_shot in shots:
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
    return few_shot_scenarios


# def run_mean_representation_on_one_scenario(scenario, models, style_per_model, recall_precisions, confusion_mat):
#     class_representations = {}
#     for style in classes:
#         class_representations.setdefault(style, [])
#         for example_model in scenario['examples'][style]:
#             filename = models[example_model]
#             class_representations[style] += [np.load(filename)]
#         class_representations[style] = np.array(class_representations[style])
#         class_representations[style] = np.mean(class_representations[style], 0)
#
#     class_vectors = np.array(list(class_representations.values()))
#
#     test_vectors = []
#     targets = []
#     for test_model in scenario['test']:
#         test_vector = np.load(models[test_model])
#         test_vectors.append(test_vector)
#         targets.append(classes.index(style_per_model[test_model]))
#
#     test_vectors = np.array(test_vectors)
#     distances = cdist(test_vectors, class_vectors)
#     predictions = np.argmax(distances, 1)
#
#     metrics = precision_recall(targets, predictions, classes)
#     for metric in metrics:
#         if metric in recall_precisions:
#             recall_precisions[metric].append(metrics[metric])
#     cmn = get_normalized_confusion_matrix(targets, predictions, classes)
#     confusion_mat += cmn


def run_knn_on_one_scenario(scenario, models, style_per_model, recall_precisions, confusion_mat):
    example_vectors = []
    example_classes = []

    for style in classes:
        for example_model in scenario['examples'][style]:
            filename = models[example_model]
            example_vectors += [np.load(filename)]
            example_classes += [classes.index(style)]

    example_vectors = np.array(example_vectors)
    example_classes = np.array(example_classes)
    class_tree = spatial.cKDTree(example_vectors)

    test_vectors = []
    targets = []
    for test_model in scenario['test']:
        test_vector = np.load(models[test_model])
        test_vectors.append(test_vector)
        targets.append(classes.index(style_per_model[test_model]))

    test_vectors = np.array(test_vectors)
    distances, knns = class_tree.query(test_vectors)
    predictions = example_classes[knns]

    metrics = precision_recall(targets, predictions, classes)
    for metric in metrics:
        if metric in recall_precisions:
            recall_precisions[metric].append(metrics[metric])
    cmn = get_normalized_confusion_matrix(targets, predictions, classes)
    confusion_mat += cmn


def run_svm_on_one_scenario(scenario, models, style_per_model, recall_precisions, confusion_mat):
    example_vectors = []
    example_classes = []

    for style in classes:
        for example_model in scenario['examples'][style]:
            filename = models[example_model]
            example_vectors += [np.load(filename)]
            example_classes += [classes.index(style)]

    example_vectors = np.array(example_vectors)
    example_classes = np.array(example_classes)

    test_vectors = []
    targets = []
    for test_model in scenario['test']:
        test_vector = np.load(models[test_model])
        test_vectors.append(test_vector)
        targets.append(classes.index(style_per_model[test_model]))

    test_vectors = np.array(test_vectors)

    clf = svm.SVC(class_weight='balanced')
    clf.fit(example_vectors, example_classes)

    predictions = clf.predict(test_vectors)

    metrics = precision_recall(targets, predictions, classes)
    for metric in metrics:
        if metric in recall_precisions:
            recall_precisions[metric].append(metrics[metric])
    cmn = get_normalized_confusion_matrix(targets, predictions, classes)
    confusion_mat += cmn


def write_result_kshot(few_shot, out_dir, recall_precisions, confusion_mat):
    print(f"{few_shot}-shot: \n")
    with open(f"{out_dir}/{few_shot}-shot-eval.txt", "w") as fout:
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
    show_confusion_matrix(confusion_mat, classes, "", f"{out_dir}/{few_shot}-shot-cmat.jpg")


def run_knn():
    for few_shot, scenarios in few_shot_scenarios.items():
        confusion_mat = np.zeros((len(classes), len(classes)))
        recall_precisions = {}
        for style in classes:
            recall_precisions[f'Recall{style}'] = []
            recall_precisions[f'Precision{style}'] = []

        for scenario_id, scenario in scenarios.items():
            run_knn_on_one_scenario(scenario, models, style_per_model, recall_precisions, confusion_mat)

        os.makedirs(os.path.join(out_dir, "classification_few_shot_knn"), exist_ok=True)
        write_result_kshot(few_shot, os.path.join(out_dir, "classification_few_shot_knn"), recall_precisions,
                           confusion_mat)


def run_svm():
    for few_shot, scenarios in few_shot_scenarios.items():
        confusion_mat = np.zeros((len(classes), len(classes)))
        recall_precisions = {}
        for style in classes:
            recall_precisions[f'Recall{style}'] = []
            recall_precisions[f'Precision{style}'] = []

        for scenario_id, scenario in scenarios.items():
            run_svm_on_one_scenario(scenario, models, style_per_model, recall_precisions, confusion_mat)

        os.makedirs(os.path.join(out_dir, "classification_few_shot_svm"), exist_ok=True)
        write_result_kshot(few_shot, os.path.join(out_dir, "classification_few_shot_svm"), recall_precisions,
                           confusion_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encodings_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--scenarios", required=True, type=str, help="pkl file")
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--exclude", required=True, type=str)
    parser.add_argument("--models_dir", required=True, type=str)
    parser.add_argument("--classes", required=True, type=str, help="e.g. C1,C2,C3,C4,C5,C6,C7,C8")
    parser.add_argument("--shots", default="5,10,20", type=str)
    args = parser.parse_args()

    classes = args.classes.split(",")
    out_dir = args.out_dir
    encodings_dir = args.encodings_dir
    models_dir = args.models_dir
    expected = args.expected
    exclude = args.exclude.split(",")
    shots = [int(s) for s in args.shots.split(",")]

    models_per_style, style_per_model, models = get_data()
    few_shot_scenarios = get_few_shot_scenarios(models_per_style, models)

    if os.path.exists(args.scenarios):
        with open(args.scenarios, "rb") as fin:
            few_shot_scenarios = pickle.load(fin)
    else:
        os.makedirs(os.path.dirname(args.scenarios), exist_ok=True)
        with open(args.scenarios, "wb") as fout:
            pickle.dump(few_shot_scenarios, fout)

    run_knn()
    run_svm()
