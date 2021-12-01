import json
import logging
from os.path import join, dirname

import numpy as np

_THR_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THR_TOL_64 = 2.0 * np.finfo(np.float64).eps

RESOURCES_DIR = join(dirname(dirname(__file__)), 'resources')
STYLES_TXT = join(RESOURCES_DIR, 'STYLES.txt')
with open(STYLES_TXT, "r") as fin:
    lines = fin.readlines()
    STYLES = [line.strip() for line in lines]
    # print("STYLES: {}".format(STYLES))

ANNFASS_STYLISTIC_ELEMENTS_JSON = join(RESOURCES_DIR, 'ANNFASS_STYLISTIC_ELEMENTS.json')
with open(ANNFASS_STYLISTIC_ELEMENTS_JSON, "r") as fin:
    ANNFASS_STYLISTIC_ELEMENTS = json.load(fin)

BUILDNET_ELEMENTS_JSON = join(RESOURCES_DIR, 'BUILDNET_ELEMENTS.json')
with open(BUILDNET_ELEMENTS_JSON, "r") as fin:
    BUILDNET_ELEMENTS = json.load(fin)

BUILDNET_STYLISTIC_ELEMENTS_JSON = join(RESOURCES_DIR, 'BUILDNET_STYLISTIC_ELEMENTS_v2.json')
with open(BUILDNET_STYLISTIC_ELEMENTS_JSON, "r") as fin:
    BUILDNET_STYLISTIC_ELEMENTS = json.load(fin)

BUILDNET_ELEMENTS_TO_REMOVE_JSON = join(RESOURCES_DIR, 'BUILDNET_ELEMENTS_TO_REMOVE.json')
with open(BUILDNET_ELEMENTS_TO_REMOVE_JSON, "r") as fin:
    BUILDNET_ELEMENTS_TO_REMOVE = json.load(fin)


class RunningBlenderCodeOutsideBlender(Exception):
    def __init__(self, message):
        super().__init__(message)


def str2bool(v):
  return v.lower() in ('true', '1')


def set_logger_file(log_file, logger):
    file_handler = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            logger.removeHandler(hdlr)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def parse_buildings_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";")[1])
    print("buildings to process: {}".format(buildings))
    return buildings


def parse_buildings_with_style_csv(filename):
    buildings = []
    with open(filename, "r") as f:
        for line in f:
            buildings.append(line.strip().split(";"))
    print("buildings to process: {}".format(buildings))
    return buildings


def parse_components_with_style_csv(filename):
    return parse_buildings_with_style_csv(filename)


class UndirectedGraph:

    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def depth_first_search(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True
        # Store the vertex to list
        temp.append(v)
        # Repeat for all vertices adjacent to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.depth_first_search(temp, i, visited)
        return temp

    def add_edge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    def connected_components(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.depth_first_search(temp, v, visited))
        return cc
