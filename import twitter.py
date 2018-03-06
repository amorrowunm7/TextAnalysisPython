import numpy as np
import igraph
import pickle
import requests
import boilerpipe
import opengraph
import pprint
import csv

# Hide some silly output
import logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Import everything we need from graphlab
import graphlab as gl


clusterGraph = igraph.Graph()

# List of vertex names
vertices = range(10)

# Add vertices to the graph
clusterGraph.add_vertices(len(vertices))
clusterGraph.vs["name"] = vertices

# Setup edges
edges = [ (0, 1), (0,2), (1, 2), (2, 3), (1,3),  (4,0), (3, 4), (1, 4), (4, 5), (5, 6), (6, 7), (7,8), (6,8), (6, 9), (8, 9), (7,9) ]
clusterGraph.add_edges( edges )

# Display (popup, since Cairo is a pain to install)
layout = clusterGraph.layout("graphopt")
igraph.plot(clusterGraph, layout = layout, vertex_label=vertices)

distances = clusterGraph.shortest_paths_dijkstra()
distances