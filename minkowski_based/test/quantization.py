import numpy as np
# import MinkowskiEngineBackend as MEB
import MinkowskiEngine as ME


N = 10
ignore_label = 255


coords = (np.random.rand(N, 4) * 100).astype(np.int32)
feats = np.random.rand(N, 4)
labels = np.floor(np.random.rand(N) * 3)

labels = labels.astype(np.int32)

# Make duplicates
coords[:3] = 0
labels[:3] = 2
print(labels)

coords_aug, feats, labels = ME.utils.sparse_quantize(coords, feats, labels=labels, ignore_label=ignore_label)
print('Unique labels and counts:',  np.unique(labels, return_counts=True))