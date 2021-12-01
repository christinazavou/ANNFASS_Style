from plyfile import PlyData
import numpy as np


plydata = PlyData.read("/home/graphicslab/Desktop/predict_on_val/pred_0000.ply")
data = plydata.elements[0].data
coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
print(coords.min(0), coords.max(0))
