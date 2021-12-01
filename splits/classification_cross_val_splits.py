import argparse
import itertools
import json
import os
import random
import sys
from ast import literal_eval
import logging

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES, set_logger_file

LOGGER = logging.getLogger(__file__)


def clean_and_init(df):
    df = df[:-2]  # remove rows "Count" and "Total"
    for col in df.columns:
        if col in IGNORE_STYLES:
            df = df.drop(columns=[col])
        if any(s+'.1' == col for s in IGNORE_STYLES):
            df = df.drop(columns=[col])
        # keep only columns <style>.1 which represent amount of unique components in <style>
        if col in STYLES:
            df = df.drop(columns=[col])
    component_style_columns_indices = []
    for idx, col in enumerate(df.columns):
        if 'building' not in col:
            component_style_columns_indices.append(idx)
    LOGGER.info("component_style_columns_indices: {}".format(component_style_columns_indices))
    print("component_style_columns_indices: {}".format(component_style_columns_indices))

    buildings_w_idx = [(row['building'], idx) for idx, row in df.iterrows()]
    styles = list(df.columns[component_style_columns_indices])
    assert set(styles) == {s+'.1' for s in STYLES}
    LOGGER.info("{} buildings: {}".format(len(buildings_w_idx), buildings_w_idx))
    print("{} buildings: {}".format(len(buildings_w_idx), buildings_w_idx))
    LOGGER.info("{} styles: {}".format(len(styles), styles))
    print("{} styles: {}".format(len(styles), styles))

    return df, component_style_columns_indices, buildings_w_idx, styles


def get_per_style_data(df, styles):
    print(f"total #components per building:\n{df[styles].sum(axis=1, skipna=True)}")
    print(f"total #components per style:\n{df[styles].sum(axis=0, skipna=True)}")

    print(f"minimum #components for one building per style:\n{df[df[styles]>0][styles].min(axis=0, skipna=True)}")
    print(f"maximum #components for one building per style:\n{df[df[styles]>0][styles].max(axis=0, skipna=True)}")
    print(f"median #components for one building per style:\n{df[df[styles]>0][styles].median(axis=0, skipna=True)}")

    buildings_per_style = {}  # which buildings have at least one component of that style
    for style in styles:
        buildings_per_style[style] = (df[df[style] > 0].index.values)
    print("buildings per style: {}".format(buildings_per_style))
    return buildings_per_style


def sort_per_style_data(buildings_per_style):
    b_s = [(s, len(b), b) for s, b in buildings_per_style.items()]
    sorted_b_s = sorted(b_s, key=lambda x: x[1])
    print("sorted styles: {}".format(sorted_b_s))
    return sorted_b_s


class CombinationFinder:

    checked_combinations = 0
    split_rows = []

    def make_split_stats_row(self, train_df, test_df, component_style_columns_indices, buildings_w_idx, buildings_per_style):
        total_train_components = np.sum(train_df.iloc[:, component_style_columns_indices].values)
        total_test_components = np.sum(test_df.iloc[:, component_style_columns_indices].values)

        total_train_components_per_style = np.sum(train_df.iloc[:, component_style_columns_indices].values, 0)
        total_test_components_per_style = np.sum(test_df.iloc[:, component_style_columns_indices].values, 0)

        train_component_pct_per_style = np.round(
            np.true_divide(total_train_components_per_style, total_train_components), 3)
        test_component_pct_per_style = np.round(
            np.true_divide(total_test_components_per_style, total_test_components), 3)

        # notice that indices are always sorted
        train_indices = train_df.index.values
        test_indices = test_df.index.values

        # number of buildings per style
        s_b_train, s_b_test = {}, {}
        for style in buildings_per_style.keys():  # this is sorted in same way as styles
            s_b_train[style] = len([b for b in buildings_per_style[style] if b in train_indices])
            s_b_test[style] = len([b for b in buildings_per_style[style] if b in test_indices])

        split_row = [
                        [b for (b, i) in buildings_w_idx if i in train_indices],
                        [b for (b, i) in buildings_w_idx if i in test_indices],
                        len(train_indices), len(test_indices),
                        total_train_components, total_test_components,
                    ] \
                    + list(total_train_components_per_style) \
                    + list(total_test_components_per_style) \
                    + list(train_component_pct_per_style) \
                    + list(test_component_pct_per_style) \
                    + list(s_b_train.values()) \
                    + list(s_b_test.values())
        return split_row

    def check_and_add(self, final_df, component_style_columns_indices, buildings_w_idx, buildings_per_style):
        train_df = final_df[final_df['Train'] == 1]
        test_df = final_df[final_df['Train'] == 0]
        train_indices = train_df.index.values
        test_indices = test_df.index.values
        for s, style_buildings in buildings_per_style.items():
            if not any(test_idx in style_buildings for test_idx in test_indices):
                return  # violates the condition that at least one building of each style exists in test data
        # doesn't violate condition that at least one building of each style exist in test data
        for s, style_buildings in buildings_per_style.items():
            if not any(train_idx in style_buildings for train_idx in train_indices):
                return  # violates the condition that at least one unique component of each style exists in train data
        # doesn't violate condition that at least one unique component of each style exist in train data
        self.split_rows.append(self.make_split_stats_row(train_df, test_df, component_style_columns_indices, buildings_w_idx, buildings_per_style))

    def make_split(self, previous_df, remaining_sorted_b_s, component_style_columns_indices, buildings_w_idx, buildings_per_style):
        if len(remaining_sorted_b_s) == 0:
            self.check_and_add(previous_df, component_style_columns_indices, buildings_w_idx, buildings_per_style)
            return
        s, b_cnt, current_style_building_indices = remaining_sorted_b_s[0]
        del remaining_sorted_b_s[0]
        # assuming one shot then one component in training is ok
        len_current_style_building_indices = len(current_style_building_indices)
        for picked_amount in range(max(1, int(len_current_style_building_indices*0.5)), int(len_current_style_building_indices*0.8)+1):
            combos = list(itertools.combinations(current_style_building_indices, picked_amount))
            random.shuffle(combos)
            combos = combos[:args.max_combos]
            for picked_indices in combos:
                copy_df = previous_df.copy()
                for picked_idx in picked_indices:
                    copy_df.at[picked_idx, 'Train'] = 1
                self.make_split(copy_df, remaining_sorted_b_s, component_style_columns_indices, buildings_w_idx, buildings_per_style)
                self.checked_combinations += 1

    def run(self, csv_file, df_initial, styles_initial):
        if os.path.exists(csv_file):
            df_splits = pd.read_csv(csv_file, index_col=0,
                                    converters={"train_buildings": literal_eval, "test_buildings": literal_eval})
        else:
            buildings_per_style_init = get_per_style_data(df_initial, styles_initial)
            sorted_b_s_init = sort_per_style_data(buildings_per_style_init)
            df_initial['Train'] = 0
            self.make_split(df_initial, sorted_b_s_init, component_style_columns_indices_init,
                            buildings_w_idx_init, buildings_per_style_init)
            print("Checked {} combinations and kept {} of them as candidates.".format(self.checked_combinations,
                                                                                      len(self.split_rows)))
            df_splits = make_splits_df(styles_initial, combo_finder.split_rows)
            df_splits.to_csv(csv_file)
        return df_splits


def save_splits(splits_df):
    splits_df = splits_df.reset_index(drop=True)

    splits_df.to_csv(out_csv, index=True)

    annfass_splits = {}
    for idx, row in splits_df.iterrows():
        annfass_splits[idx] = {
            'train_buildings': row['test_buildings'],  # not a mistake
            'test_buildings': row['train_buildings'],  # not a mistake
        }
    with open(out_json, "w") as f:
        json.dump(annfass_splits, f, indent=4)


def make_splits_df(styles, split_rows):
    columns = ['train_buildings', 'test_buildings',
               '#train_buildings', '#test_buildings',
               '#train_components', '#test_components']
    re_order_columns = columns.copy()
    for style in styles:
        columns += ["{}_train_components".format(style)]
    for style in styles:
        columns += ["{}_test_components".format(style)]
    for style in styles:
        columns += ["{}_train_components_pct".format(style)]
    for style in styles:
        columns += ["{}_test_components_pct".format(style)]
    for style in styles:
        columns += ["{}_#train_buildings".format(style)]
    for style in styles:
        columns += ["{}_#test_buildings".format(style)]
    df = pd.DataFrame(data=split_rows, columns=columns)
    for style in styles:
        re_order_columns += [
            "{}_train_components".format(style),
            "{}_test_components".format(style),
            "{}_train_components_pct".format(style),
            "{}_test_components_pct".format(style),
            "{}_#train_buildings".format(style),
            "{}_#test_buildings".format(style)
        ]
    df = df[re_order_columns]
    return df


class SplitsSelector:

    def train_components_majority(self, df, styles, tolerance=2):
        cols = []
        for style in styles:
            df['TestCGtTrainC{}'.format(style)] = df['{}_test_components'.format(style)] > df['{}_train_components'.format(style)]
            cols += ['TestCGtTrainC{}'.format(style)]
        df['SumTestCGtTrainC'] = df[cols].sum(axis=1)

        df = df[df['SumTestCGtTrainC'] >= len(styles) - tolerance]
        for style in styles:
            df = df.drop(['TestCGtTrainC{}'.format(style)], axis=1)
        LOGGER.info(f"Kept {df.shape[0]} with #test > #train in at least {len(styles) - tolerance} classes")
        print(f"Kept {df.shape[0]} with #test > #train in at least {len(styles) - tolerance} classes")

        df['TestCGeTrainC'] = df['#test_components'] >= df['#train_components']
        df = df[df['TestCGeTrainC']]
        df = df.drop(['TestCGeTrainC'], axis=1)
        LOGGER.info("Kept {} with #test_components >= #train_components".format(df.shape[0]))
        print("Kept {} with #test_components >= #train_components".format(df.shape[0]))
        return df

    def train_buildings_majority(self, df):
        df['TestBGeTestB'] = df['#test_buildings'] >= df['#train_buildings']  # test and train are swapped
        df = df[df['TestBGeTestB']]
        df = df.drop(['TestBGeTestB'], axis=1)
        LOGGER.info("Kept {} with #test_buildings >= #train_buildings".format(df.shape[0]))
        print("Kept {} with #test_buildings >= #train_buildings".format(df.shape[0]))
        df = df[df['#train_buildings'] >= args.min_test_buildings]
        LOGGER.info("Kept {} with #train_buildings>={}".format(df.shape[0], args.min_test_buildings))
        print("Kept {} with #train_buildings>={}".format(df.shape[0], args.min_test_buildings))
        return df

    def pct_order_kept(self, df, init_pcts):
        pct_cols = [style + "_test_components_pct" for style in init_pcts.index.values]

        init_order = init_pcts.sort_values(ascending=False, ).index.values
        keep_indices = []

        for idx, row in df.iterrows():
            pcts = row[pct_cols]
            order = pcts.sort_values(ascending=False, ).index.values
            order = [o.replace("_test_components_pct", "") for o in order]
            if all(init_order == order):
                # if np.allclose(init_pcts.values.astype(np.float16), pcts.values.astype(np.float16), atol=0.05):
                keep_indices += [idx]

        df = df.loc[keep_indices]
        print("Kept {} with order kept".format(df.shape[0]))
        return df

    def run(self, df, styles, pcts, nrows=12):
        # keep digits that represent unique builings, and sort them in order to find duplicate buildings sets
        df['train'] = [",".join(sorted(["".join([c for c in li if c.isdigit()]) for li in l])) for l in df['train_buildings']]
        df = df.drop_duplicates(subset=['train'])
        df = df.drop(['train'], axis=1)
        LOGGER.info("Kept {} by removing duplicates".format(df.shape[0]))
        print("Kept {} by removing duplicates".format(df.shape[0]))

        df = self.train_buildings_majority(df)
        # df = self.pct_order_kept(df, pcts)
        df = self.train_components_majority(df, styles)

        if len(df) > nrows:
            df = df.sample(n=nrows, replace=False)
        print("Kept selection of {} spltis".format(df.shape[0]))

        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_stats_file", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--skip_rows", default="[1]", type=str)
    parser.add_argument("--ignore_styles", default="Modernist,Pagoda,Renaissance,Russian,Venetian,Unknown", type=str)
    parser.add_argument("--max_combos", default=100, type=int)
    parser.add_argument("--min_test_buildings", default=7, type=int)
    parser.add_argument("--num_splits", default=5, type=int)
    parser.add_argument("--num_repeats", default=5, type=int)
    args = parser.parse_args()

    _log_file = os.path.join(args.out_dir, "classification_cross_val_splits.log")
    LOGGER = set_logger_file(_log_file, LOGGER)
    LOGGER.info(args)

    FNAME = "classification_cross_val"

    IGNORE_STYLES = args.ignore_styles.split(",")
    STYLES = list(set(STYLES) - set(IGNORE_STYLES))

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_csv = os.path.join(args.out_dir, "{}.tmp.csv".format(FNAME))

    df_init = pd.read_csv(args.style_stats_file, sep=" ", skiprows=eval(args.skip_rows))
    df_init, component_style_columns_indices_init, buildings_w_idx_init, styles_init = clean_and_init(df_init)
    pct_init = df_init[df_init[styles_init] > 0][styles_init].sum(axis=0, skipna=True) / \
               df_init[df_init[styles_init] > 0][styles_init].sum().sum()

    combo_finder = CombinationFinder()
    df_splits = combo_finder.run(tmp_csv, df_init, styles_init)

    for repeat in range(args.num_repeats):
        os.makedirs(os.path.join(args.out_dir, f"repeat_{repeat}"), exist_ok=True)
        out_csv = os.path.join(args.out_dir, f"repeat_{repeat}", "{}.csv".format(FNAME))
        out_json = os.path.join(args.out_dir, f"repeat_{repeat}", "{}.json".format(FNAME))
        selector = SplitsSelector()
        df_splits_repeat = selector.run(df_splits, styles_init, pct_init, args.num_splits)
        save_splits(df_splits_repeat)

