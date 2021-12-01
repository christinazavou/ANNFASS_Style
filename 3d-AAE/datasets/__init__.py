from datasets import buildingcomponent
from datasets import contentstylecomponent
from datasets import modelnet40
from datasets import shapenet

MODULES = [buildingcomponent,
           contentstylecomponent,
           modelnet40,
           shapenet]

DATASETS = []
for module in MODULES:
    DATASETS += [getattr(module, a) for a in dir(module) if 'Dataset' in a]


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
