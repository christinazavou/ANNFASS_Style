import argparse
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from svm.methods.svm import SVM as SVMSimple

from common.utils import STYLES as classes

IGNORE_CLASSES = "Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown".split(",")


def run(test_file, model_pkl_file, svm_impl, out_dir):

    assert os.path.exists(test_file)

    if svm_impl == "simple":
        impl = SVMSimple
        named_args = {'ignore_classes': IGNORE_CLASSES}
    else:
        raise Exception(f"unknown implementation {svm_impl}")
    result = impl(None, test_file, classes, model_pkl_file, **named_args).infer()
    result = {name: classes[prediction] for name, prediction in result.items()}

    os.makedirs(out_dir, exist_ok=True)
    buildings = set([name.split("/")[0] for name in result])

    for building in buildings:
        building_result = {name.split("/")[1]: prediction for name, prediction in result.items()
                           if name.split("/")[0] == building}
        out_file = os.path.join(out_dir, building+".json")
        with open(out_file, "w") as fout:
            json.dump(building_result, fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--model_pkl_file", required=True, type=str)
    parser.add_argument("--svm_impl", default="simple", type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    run(args.test_csv, args.model_pkl_file, args.svm_impl, args.out_dir)
