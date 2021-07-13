"""
vertices_graph.py: Compute the polar coordinates of all patches. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

import networkx as nx
import numpy as np
import scipy.linalg
from  numpy.linalg import norm
import pymesh

def vertices_graph(mesh, weighted=False):
    # Graph 
    G = nx.Graph()
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))

    # Get edges
    f = np.array(mesh.faces, dtype = int)
    rowi = np.concatenate([f[:,0], f[:,0], f[:,1], f[:,1], f[:,2], f[:,2]], axis = 0)
    rowj = np.concatenate([f[:,1], f[:,2], f[:,0], f[:,2], f[:,0], f[:,1]], axis = 0)

    if(weighted):
        #verts = mesh.vertices
        # Get weights 
        edgew = mesh.vertices[rowi] - mesh.vertices[rowj]
        edgew = scipy.linalg.norm(edgew, axis=1)
        wedges = np.stack([rowi, rowj, edgew]).T
        G.add_weighted_edges_from(wedges)
    else:
        edges = np.stack([rowi, rowj]).T
        G.add_edges_from(edges)
        
    return G
 
