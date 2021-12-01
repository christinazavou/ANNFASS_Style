from datasets.Component import ComponentObjDataset, ComponentMeshDataset, ComponentSamplesDataset
from datasets.dataset_utils import collate_pointcloud_fn
from datasets.dataset_utils import collate_content_style_cloud_fn
from datasets.dataset_utils import collate_pointcloud_with_features_fn
from datasets.dataset_utils import collate_content_style_cloud_with_features_fn

MODULES = [
              ComponentObjDataset,
              ComponentMeshDataset,
              ComponentSamplesDataset
           ]

DATASETS = []
for module in MODULES:
    DATASETS += [getattr(module, a) for a in dir(module) if 'Dataset' in a]

COLLATE_FNS = [collate_pointcloud_fn,
               collate_content_style_cloud_fn,
               collate_pointcloud_with_features_fn,
               collate_content_style_cloud_with_features_fn]


def load_dataset_class(name):
    mdict = {dataset.__name__: dataset for dataset in DATASETS}
    if name not in mdict:
        print('Invalid dataset index. Options are:')
        # Display a list of valid dataset names
        for dataset in DATASETS:
            print('\t* {}'.format(dataset.__name__))
        raise ValueError(f'Dataset {name} not defined')
    DatasetClass = mdict[name]
    return DatasetClass


def load_collate_fn(name):
    mdict = {fn.__name__: fn for fn in COLLATE_FNS}
    if name not in mdict:
        print('Invalid dataset index. Options are:')
        # Display a list of valid dataset names
        for fn in COLLATE_FNS:
            print('\t* {}'.format(fn.__name__))
        raise ValueError(f'Collate_fn {name} not defined')
    collate_function = mdict[name]
    return collate_function
