from unittest import TestCase
from lib.utils import precision_at_one, calculate_iou, calculate_iou_custom, get_prediction, \
    get_with_component_criterion, class_balanced, inverse_freq, get_class_frequencies
import numpy as np
import torch


class Test(TestCase):
    def test_precision_at_one(self):
        pred = np.array([1, 2, 3, 1, 2, 2, 2, 3, 3])
        targ = np.array([1, 2, 1, 1, 1, 2, 2, 1, 3])
        pred = torch.from_numpy(pred).to('cpu')
        targ = torch.from_numpy(targ).to('cpu')
        res = precision_at_one(pred, targ)
        print(res)
        assert np.isclose(res, 66.666, atol=0.001)

        res = precision_at_one(pred, targ, ignore_label=3)
        assert np.equal(res, 62.5)

    def test_calculate_iou(self):
        pred = np.array([1, 2, 3, 1, 2, 2, 2, 3, 3])
        targ = np.array([1, 2, 1, 1, 1, 2, 2, 1, 3])

        res = calculate_iou(targ, pred, 3)
        print(res)
        res = calculate_iou_custom(targ, pred, 4, 0)
        print(res)

    def test_get_prediction(self):
        logits = np.random.random((5, 4))
        logits = torch.from_numpy(logits)
        res = get_prediction(None, logits, None)
        print(res)

    # def test_get_cls_criterion(self):
    #     cr = get_cls_criterion(None)
    #     print(cr.__class__.__name__)


def test_class_weighting():
    print("Split fold0\n")
    # str_class_counts: {'ignore': 127, 'Unknown': 111, 'Colonial': 174, 'Neo_classicism': 727, 'Modernist': 191, 'Ottoman': 32, 'Gothic': 10, 'Byzantine': 25, 'Venetian': 44, 'Baroque': 103, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 429, 0: 174, 1: 727, 2: 32, 3: 10, 4: 25, 5: 44, 6: 103, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold1\n")
    # str_class_counts: {'ignore': 161, 'Unknown': 71, 'Colonial': 221, 'Neo_classicism': 299, 'Modernist': 230, 'Ottoman': 80, 'Gothic': 5, 'Byzantine': 13, 'Venetian': 27, 'Baroque': 141, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 462, 0: 221, 1: 299, 2: 80, 3: 5, 4: 13, 5: 27, 6: 141, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold2\n")
    # str_class_counts: {'ignore': 114, 'Unknown': 169, 'Colonial': 135, 'Neo_classicism': 908, 'Modernist': 195, 'Ottoman': 82, 'Gothic': 10, 'Byzantine': 35, 'Venetian': 22, 'Baroque': 134, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 478, 0: 135, 1: 908, 2: 82, 3: 10, 4: 35, 5: 22, 6: 134, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold3\n")
    # str_class_counts: {'ignore': 191, 'Unknown': 124, 'Colonial': 56, 'Neo_classicism': 754, 'Modernist': 365, 'Ottoman': 83, 'Gothic': 10, 'Byzantine': 9, 'Venetian': 46, 'Baroque': 81, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 680, 0: 56, 1: 754, 2: 83, 3: 10, 4: 9, 5: 46, 6: 81, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold4\n")
    # str_class_counts: {'ignore': 136, 'Unknown': 143, 'Colonial': 344, 'Neo_classicism': 647, 'Modernist': 240, 'Ottoman': 32, 'Gothic': 5, 'Byzantine': 12, 'Venetian': 42, 'Baroque': 129, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 519, 0: 344, 1: 647, 2: 32, 3: 5, 4: 12, 5: 42, 6: 129, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold5\n")
    # str_class_counts: {'ignore': 17, 'Unknown': 130, 'Colonial': 178, 'Neo_classicism': 737, 'Modernist': 354, 'Ottoman': 81, 'Gothic': 4, 'Byzantine': 31, 'Venetian': 26, 'Baroque': 59, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 501, 0: 178, 1: 737, 2: 81, 3: 4, 4: 31, 5: 26, 6: 59, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold6\n")
    # str_class_counts: {'ignore': 208, 'Unknown': 78, 'Colonial': 225, 'Neo_classicism': 616, 'Modernist': 301, 'Ottoman': 83, 'Gothic': 10, 'Byzantine': 34, 'Venetian': 50, 'Baroque': 75, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 587, 0: 225, 1: 616, 2: 83, 3: 10, 4: 34, 5: 50, 6: 75, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold7\n")
    # str_class_counts: {'ignore': 215, 'Unknown': 130, 'Colonial': 223, 'Neo_classicism': 699, 'Modernist': 341, 'Ottoman': 32, 'Gothic': 5, 'Byzantine': 8, 'Venetian': 46, 'Baroque': 82, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 686, 0: 223, 1: 699, 2: 32, 3: 5, 4: 8, 5: 46, 6: 82, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold8\n")
    # str_class_counts: {'ignore': 84, 'Unknown': 189, 'Colonial': 309, 'Neo_classicism': 689, 'Modernist': 341, 'Ottoman': 82, 'Gothic': 4, 'Byzantine': 11, 'Venetian': 26, 'Baroque': 117, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 614, 0: 309, 1: 689, 2: 82, 3: 4, 4: 11, 5: 26, 6: 117, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold9\n")
    # str_class_counts: {'ignore': 177, 'Unknown': 140, 'Colonial': 287, 'Neo_classicism': 814, 'Modernist': 347, 'Ottoman': 81, 'Gothic': 0, 'Byzantine': 24, 'Venetian': 44, 'Baroque': 109, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 664, 0: 287, 1: 814, 2: 81, 3: 0, 4: 24, 5: 44, 6: 109, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold10\n")
    # str_class_counts: {'ignore': 65, 'Unknown': 23, 'Colonial': 356, 'Neo_classicism': 395, 'Modernist': 335, 'Ottoman': 52, 'Gothic': 6, 'Byzantine': 25, 'Venetian': 45, 'Baroque': 86, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 423, 0: 356, 1: 395, 2: 52, 3: 6, 4: 25, 5: 45, 6: 86, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})
    print("Split fold11\n")
    # str_class_counts: {'ignore': 108, 'Unknown': 141, 'Colonial': 167, 'Neo_classicism': 363, 'Modernist': 220, 'Ottoman': 83, 'Gothic': 5, 'Byzantine': 32, 'Venetian': 48, 'Baroque': 129, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    final_class_counts = {-1: 469, 0: 167, 1: 363, 2: 83, 3: 5, 4: 32, 5: 48, 6: 129, 7: 0, 8: 0}
    frequencies = get_class_frequencies(final_class_counts, -1)
    print({l: round(f, 3) for l, f in frequencies.items()})
    weights = class_balanced(frequencies, 0.99999)
    print({l: round(w, 3) for l, w in weights.items()})
    weights = inverse_freq(frequencies)
    print({l: round(w, 3) for l, w in weights.items()})

    # Val splits:
    # Split fold0
    # str_class_counts: {'ignore': 85, 'Unknown': 40, 'Colonial': 167, 'Neo_classicism': 133, 'Modernist': 219, 'Ottoman': 50, 'Gothic': 0, 'Byzantine': 9, 'Venetian': 0, 'Baroque': 27, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 344, 0: 167, 1: 133, 2: 50, 3: 0, 4: 9, 5: 0, 6: 27, 7: 0, 8: 0}
    # Split fold1
    # str_class_counts: {'ignore': 53, 'Unknown': 102, 'Colonial': 110, 'Neo_classicism': 623, 'Modernist': 184, 'Ottoman': 2, 'Gothic': 5, 'Byzantine': 22, 'Venetian': 22, 'Baroque': 0, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 339, 0: 110, 1: 623, 2: 2, 3: 5, 4: 22, 5: 22, 6: 0, 7: 0, 8: 0}
    # Split fold10
    # str_class_counts: {'ignore': 149, 'Unknown': 125, 'Colonial': 4, 'Neo_classicism': 585, 'Modernist': 77, 'Ottoman': 30, 'Gothic': 5, 'Byzantine': 7, 'Venetian': 0, 'Baroque': 38, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 351, 0: 4, 1: 585, 2: 30, 3: 5, 4: 7, 5: 0, 6: 38, 7: 0, 8: 0}
    # Split fold11
    # str_class_counts: {'ignore': 105, 'Unknown': 73, 'Colonial': 189, 'Neo_classicism': 750, 'Modernist': 190, 'Ottoman': 0, 'Gothic': 5, 'Byzantine': 1, 'Venetian': 0, 'Baroque': 0, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 368, 0: 189, 1: 750, 2: 0, 3: 5, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
    # Split fold2
    # str_class_counts: {'ignore': 101, 'Unknown': 13, 'Colonial': 189, 'Neo_classicism': 93, 'Modernist': 225, 'Ottoman': 0, 'Gothic': 0, 'Byzantine': 0, 'Venetian': 27, 'Baroque': 4, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 339, 0: 189, 1: 93, 2: 0, 3: 0, 4: 0, 5: 27, 6: 4, 7: 0, 8: 0}
    # Split fold3
    # str_class_counts: {'ignore': 21, 'Unknown': 85, 'Colonial': 295, 'Neo_classicism': 223, 'Modernist': 45, 'Ottoman': 0, 'Gothic': 0, 'Byzantine': 24, 'Venetian': 0, 'Baroque': 46, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 151, 0: 295, 1: 223, 2: 0, 3: 0, 4: 24, 5: 0, 6: 46, 7: 0, 8: 0}
    # Split fold4
    # str_class_counts: {'ignore': 77, 'Unknown': 82, 'Colonial': 0, 'Neo_classicism': 143, 'Modernist': 179, 'Ottoman': 50, 'Gothic': 5, 'Byzantine': 22, 'Venetian': 0, 'Baroque': 4, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 338, 0: 0, 1: 143, 2: 50, 3: 5, 4: 22, 5: 0, 6: 4, 7: 0, 8: 0}
    # Split fold5
    # str_class_counts: {'ignore': 193, 'Unknown': 8, 'Colonial': 164, 'Neo_classicism': 140, 'Modernist': 62, 'Ottoman': 0, 'Gothic': 6, 'Byzantine': 2, 'Venetian': 22, 'Baroque': 74, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 263, 0: 164, 1: 140, 2: 0, 3: 6, 4: 2, 5: 22, 6: 74, 7: 0, 8: 0}
    # Split fold6
    # str_class_counts: {'ignore': 4, 'Unknown': 94, 'Colonial': 115, 'Neo_classicism': 265, 'Modernist': 110, 'Ottoman': 0, 'Gothic': 0, 'Byzantine': 0, 'Venetian': 0, 'Baroque': 54, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 208, 0: 115, 1: 265, 2: 0, 3: 0, 4: 0, 5: 0, 6: 54, 7: 0, 8: 0}
    # Split fold7
    # str_class_counts: {'ignore': 0, 'Unknown': 50, 'Colonial': 120, 'Neo_classicism': 78, 'Modernist': 71, 'Ottoman': 51, 'Gothic': 4, 'Byzantine': 23, 'Venetian': 0, 'Baroque': 56, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 121, 0: 120, 1: 78, 2: 51, 3: 4, 4: 23, 5: 0, 6: 56, 7: 0, 8: 0}
    # Split fold8
    # str_class_counts: {'ignore': 131, 'Unknown': 27, 'Colonial': 29, 'Neo_classicism': 172, 'Modernist': 79, 'Ottoman': 0, 'Gothic': 5, 'Byzantine': 22, 'Venetian': 22, 'Baroque': 4, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 237, 0: 29, 1: 172, 2: 0, 3: 5, 4: 22, 5: 22, 6: 4, 7: 0, 8: 0}
    # Split fold9
    # str_class_counts: {'ignore': 31, 'Unknown': 8, 'Colonial': 56, 'Neo_classicism': 84, 'Modernist': 62, 'Ottoman': 0, 'Gothic': 8, 'Byzantine': 10, 'Venetian': 0, 'Baroque': 23, 'Russian': 0, 'Romanesque': 0, 'Renaissance': 0, 'Pagoda': 0, 'Empty': 0}
    # final_class_counts: {-1: 101, 0: 56, 1: 84, 2: 0, 3: 8, 4: 10, 5: 0, 6: 23, 7: 0, 8: 0}
