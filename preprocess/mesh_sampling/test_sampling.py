# import geometry_utils
# geometry_utils.config(DEBUG=True)
# import general_utils
# general_utils.config()
#
# import os
# import argparse
# import sys
# import numpy as np
# from collections import OrderedDict
#
# # Add igl library to the modules search path
# HOME_DIR = os.path.expanduser("~")
# LIBIGL_DIR = os.path.join(HOME_DIR, 'Repos', 'learning_segmentation_boundaries', 'external_lib', 'libigl', 'python')
# sys.path.append(LIBIGL_DIR)
# print(os.listdir(LIBIGL_DIR))
# import pyigl as igl
#
#
# def key_pressed(viewer, key, modifier):
#
# 	if key == ord('1'):
# 		viewer.data().clear()
# 		viewer.data().set_mesh(VEigen, FEigen)
# 		viewer.data().set_colors(colourComponentsEigen)
#
# 		return True
# 	if key == ord('0'):
# 		exit(-1)
#
# if __name__ == "__main__":
#
# 	# Load data
# 	partnet = []
# 	with open("/media/melinos/BigData/ShapeNetRAW/partnet_dataset/stats/Bed_train_val_test_split.txt", "rb") as f:
# 		for line in f:
# 			record = line.rstrip().split(' ')
# 			partnet.append([record[0], record[1]])
#
# 	partnet_dir = "/media/melinos/BigData/ShapeNetRAW/partnet_dataset"
# 	for modelInfo in partnet:
# 		level=3
# 		hierarchicalMesh = geometry_utils.HierarchicalMesh(category=modelInfo[0], annoID=modelInfo[1],
# 														   partnetDir=partnet_dir, process=True, meshFix=True)
#
# 		# Poisson sampling(?) using libigl
#
#
# 		# Visualize first level
# 		VEigen = general_utils.np2eigen(hierarchicalMesh.V); FEigen = general_utils.np2eigen(hierarchicalMesh.F)
# 		viewer = igl.glfw.Viewer()
# 		viewer.core.background_color = igl.eigen.MatrixXd([1, 1, 1, 1])
# 		viewer.data().set_mesh(VEigen, FEigen)
# 		colourComponentsEigen = igl.eigen.MatrixXd()
# 		componentsEigen = general_utils.np2eigen(hierarchicalMesh.getPartsFromLevel(level=level).astype(np.float32))
# 		igl.jet(componentsEigen, True, colourComponentsEigen)
# 		viewer.data().set_colors(colourComponentsEigen)
# 		viewer.callback_key_pressed = key_pressed
#
# 		viewer.launch()
#

import numpy as np
import matplotlib.pyplot as plt

# Choose up to k points around each reference point as candidates for a new
# sample point
k = 30

# Minimum distance between samples
r = 1.7

width, height = 60, 45

# Cell side length
a = r/np.sqrt(2)
# Number of cells in the x- and y-directions of the grid
nx, ny = int(width / a) + 1, int(height / a) + 1

# A list of coordinates in the grid of cells
coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
# Initilalize the dictionary of cells: each key is a cell's coordinates, the
# corresponding value is the index of that cell's point's coordinates in the
# samples list (or None if the cell is empty).
cells = {coords: None for coords in coords_list}

def get_cell_coords(pt):
    """Get the coordinates of the cell that pt = (x,y) falls in."""

    return int(pt[0] // a), int(pt[1] // a)

def get_neighbours(coords):
    """Return the indexes of points in cells neighbouring cell at coords.

    For the cell at coords = (x,y), return the indexes of points in the cells
    with neighbouring coordinates illustrated below: ie those cells that could
    contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

    """

    dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
            (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
            (-1,2),(0,2),(1,2),(0,0)]
    neighbours = []
    for dx, dy in dxdy:
        neighbour_coords = coords[0] + dx, coords[1] + dy
        if not (0 <= neighbour_coords[0] < nx and
                0 <= neighbour_coords[1] < ny):
            # We're off the grid: no neighbours here.
            continue
        neighbour_cell = cells[neighbour_coords]
        if neighbour_cell is not None:
            # This cell is occupied: store this index of the contained point.
            neighbours.append(neighbour_cell)
    return neighbours

def point_valid(pt):
    """Is pt a valid point to emit as a sample?

    It must be no closer than r from any other point: check the cells in its
    immediate neighbourhood.

    """

    cell_coords = get_cell_coords(pt)
    for idx in get_neighbours(cell_coords):
        nearby_pt = samples[idx]
        # Squared distance between or candidate point, pt, and this nearby_pt.
        distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
        if distance2 < r**2:
            # The points are too close, so pt is not a candidate.
            return False
    # All points tested: if we're here, pt is valid
    return True

def get_point(k, refpt):
    """Try to find a candidate point relative to refpt to emit in the sample.

    We draw up to k points from the annulus of inner radius r, outer radius 2r
    around the reference point, refpt. If none of them are suitable (because
    they're too close to existing points in the sample), return False.
    Otherwise, return the pt.

    """
    i = 0
    while i < k:
        rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
        pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt):
            return pt
        i += 1
    # We failed to find a suitable point in the vicinity of refpt.
    return False

# Pick a random point to start with.
pt = (np.random.uniform(0, width), np.random.uniform(0, height))
samples = [pt]
# Our first sample is indexed at 0 in the samples list...
cells[get_cell_coords(pt)] = 0
# ... and it is active, in the sense that we're going to look for more points
# in its neighbourhood.
active = [0]

nsamples = 1
# As long as there are points in the active list, keep trying to find samples.
while active:
    # choose a random "reference" point from the active list.
    idx = np.random.choice(active)
    refpt = samples[idx]
    # Try to pick a new point relative to the reference point.
    pt = get_point(k, refpt)
    if pt:
        # Point pt is valid: add it to the samples list and mark it as active
        samples.append(pt)
        nsamples += 1
        active.append(len(samples)-1)
        cells[get_cell_coords(pt)] = len(samples) - 1
    else:
        # We had to give up looking for valid points near refpt, so remove it
        # from the list of "active" points.
        active.remove(idx)

plt.scatter(*zip(*samples), color='r', alpha=0.6, lw=0)
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.show()
