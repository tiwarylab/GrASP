import numpy as np
import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
from MDAnalysis.analysis.distances import contact_matrix, distance_array
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
import networkx as nx 
from networkx.algorithms.community import louvain_communities
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import linprog
import os
from tqdm import tqdm
from glob import glob

from sklearn.metrics import roc_curve, auc 
import time

import sys
import argparse
from joblib import Parallel, delayed
import warnings


def center_of_mass(coords, masses):
    """Compute center of mass for a set of atoms.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of atomic coordinates.
    masses: numpy.ndarray
        Array of atomic masses.

    Returns
    -------
    numpy.ndarray
        Center of mass.
    """

    return np.sum(coords*np.tile(masses, (3,1)).T, axis=0)/np.sum(masses)

def sort_clusters(cluster_ids, probs, labels, score_type='mean'):
    """Sort clusters according to binding site scores.

    Parameters
    ----------
    cluster_ids : numpy.ndarray
        Cluster label for each atom.
        
    probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.
        
    labels: numpy.ndarray
        Model predicting classes (binding/non-binding) for each atom.

    score_type: str
        Option for atom score aggregation.

    Returns
    -------
    numpy.ndarray
        Resorted clusters labels with highest scoring first.
    """

    c_probs = []
    unique_ids = np.unique(cluster_ids)

    for c_id in unique_ids[unique_ids >= 0]:
        if score_type == 'mean':
            c_prob = np.mean(probs[:,1][labels][cluster_ids==c_id])
        elif score_type == 'sum':
            c_prob = np.sum(probs[:,1][labels][cluster_ids==c_id])
        elif score_type == 'square':
            c_prob = np.sum((probs[:,1][labels][cluster_ids==c_id])**2)
        else:
            print('sort_clusters score_type must be mean, sum, or square.')
        c_probs.append(c_prob)

    c_order = np.argsort(c_probs)

    sorted_ids = -1*np.ones(cluster_ids.shape)
    for new_c in range(len(c_order)):
        old_c = c_order[new_c]
        sorted_ids[cluster_ids == old_c] = new_c
        
    return sorted_ids

def cluster_atoms_meanshift(all_coords, predicted_probs, threshold=.5, quantile=.3, bw=None, score_type='mean', **kwargs):
    """Cluster atoms into sites with meanshift clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    quantile: float
        Quantile for determining meanshift bandwidth.

    bw: float
        Static meanshift bandwidth (as opposed to quantile-based).

    score_type: str
        Option for atom score aggregation.

    **kwargs
        Additional meanshift kwargs.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.

    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        if bw is None:
            bw = estimate_bandwidth(bind_coords, quantile=quantile)
        if bw == 0:
            bw = 1e-17
        try:
            ms_clustering = MeanShift(bandwidth=bw, **kwargs).fit(bind_coords)
        except Exception as e:
            raise e
        cluster_ids = ms_clustering.labels_
        
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = [0]

    all_ids = -1*np.ones(predicted_labels.shape)
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def cluster_atoms_DBSCAN(all_coords, predicted_probs, threshold=.5, eps=3, min_samples=5, score_type='mean'):
    """Cluster atoms into sites with DBSCAN clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    eps: float
        DBSCAN neighbor distance cutoff.

    min_samples: int
        Minimum neighbors for a point to be considered a core point.

    score_type: str
        Option for atom score aggregation.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        ms_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(bind_coords)
        cluster_ids = ms_clustering.labels_
        
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = [0]

    all_ids = -1*np.ones(predicted_labels.shape)
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def cluster_atoms_louvain(all_coords, adj_matrix, predicted_probs, threshold=.5, cutoff=5, resolution=0.05, score_type='mean'):
    """Cluster atoms into sites with Louvain community detection.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.

    adj_matrix: numpy.ndarray
        Distance matrix to construct the graph for community detection.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    cutoff: float
        Distance cutoff to connect atoms in the graph.

    resolution: int
        Louvain community detection resolution.

    score_type: str
        Option for atom score aggregation.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels=predicted_probs[:,1] > threshold
    
    G = nx.from_scipy_sparse_array(adj_matrix, edge_attribute="distance")
    prob_dict = {i:predicted_probs[i,1] for i in range(len(predicted_probs))}
    nx.set_node_attributes(G, prob_dict, name='probability')
    inverse_distance = nx.get_edge_attributes(G, name="distance")
    inverse_distance = {k:-1*v for k, v in inverse_distance.items()}
    nx.set_edge_attributes(G, inverse_distance, "inverse_distance")
    
    remove = [node for node,probability in G.nodes.data("probability") if probability < threshold]
    G.remove_nodes_from(remove)
    
    bind_coords = all_coords[predicted_labels]
    
    # Cleanout anything larger than our cutoff
    for u,v, dist in list(G.edges.data("distance")):
        if dist > cutoff:
            G.remove_edge(u,v)

    
    communities = louvain_communities(G, resolution=resolution, weight=None)
    
        
    assignment_dict = {}
    for i, ids in enumerate(communities):
        for id in ids:
            assignment_dict[id] = i
        
    cluster_ids = np.array([assignment_dict[k] for k in sorted(assignment_dict.keys())])
    sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    
    all_ids = -1*np.ones(predicted_labels.shape)
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def cluster_atoms_single(all_coords, predicted_probs, threshold=.5, score_type='mean', **kwargs):
    """Cluster atoms into sites with single linkage clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    score_type: str
        Option for atom score aggregation.

    **kwargs
        Additional agglomerative clustering kwargs.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """
    
    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        link_clustering = AgglomerativeClustering(linkage='single', **kwargs).fit(bind_coords)
        cluster_ids = link_clustering.labels_
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = np.ones(1)

    all_ids = -1*np.ones(predicted_labels.shape) 
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def cluster_atoms_complete(all_coords, predicted_probs, threshold=.5, score_type='mean', **kwargs):
    """Cluster atoms into sites with complete linkage clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    score_type: str
        Option for atom score aggregation.

    **kwargs
        Additional agglomerative clustering kwargs.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        link_clustering = AgglomerativeClustering(linkage='complete', **kwargs).fit(bind_coords)
        cluster_ids = link_clustering.labels_
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = np.ones(1)

    all_ids = -1*np.ones(predicted_labels.shape) 
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def cluster_atoms_average(all_coords, predicted_probs, threshold=.5, score_type='mean', **kwargs):
    """Cluster atoms into sites with average linkage clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    score_type: str
        Option for atom score aggregation.

    **kwargs
        Additional agglomerative clustering kwargs.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        link_clustering = AgglomerativeClustering(linkage='average', **kwargs).fit(bind_coords)
        cluster_ids = link_clustering.labels_
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = np.ones(1)

    all_ids = -1*np.ones(predicted_labels.shape) 
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids
    
def cluster_atoms_ward(all_coords, predicted_probs, threshold=.5, score_type='mean', **kwargs):
    """Cluster atoms into sites with ward agglomerative clustering.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    score_type: str
        Option for atom score aggregation.

    **kwargs
        Additional agglomerative clustering kwargs.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] >= threshold
    site_probs = predicted_probs[:,1]
    site_probs[site_probs < threshold] = 0
    connectivity = radius_neighbors_graph(all_coords, 5, mode='distance')
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        link_clustering = AgglomerativeClustering(connectivity=connectivity, **kwargs).fit(site_probs.reshape(-1,1))
        cluster_ids = link_clustering.labels_
        
        unique_clusters = np.unique(cluster_ids)
        exclusions = [c for c in unique_clusters if np.mean(site_probs[cluster_ids==c]) < threshold/2] # if mostly predicted as non-site, drop cluster
        for e in exclusions:
            cluster_ids[cluster_ids==e] = -1

        cluster_ids = cluster_ids[predicted_labels] # only taking points predicted as sites

        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = np.ones(1)

    all_ids = -1*np.ones(predicted_labels.shape)
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids 

def cluster_atoms_groundtruth(all_coords, lig_coord_list, predicted_probs, threshold=.5, distance_threshold=5, score_type='mean'):
    """Associate binding site atoms with the nearest ligand.

    For analyzing whether clustering or GNN is the source of error, not for method comparison.

    Parameters
    ----------
    all_coords: numpy.ndarray
        Protein atomic coordinates.

    lig_coord_list: list
        List of atomic coordinates for each ligand.
        
    predicted_probs: numpy.ndarray
        Model predicted binding site probabilities for each atom.

    threshold: float
        Probability threshold to classify atoms as site/non-site.

    distance_threshold: float
        Maximum distance to consider a site atom associated with a ligand.

    score_type: str
        Option for atom score aggregation.

    Returns
    -------
    numpy.ndarray
        Coordinates of all binding site atoms.
    
    numpy.ndarray
        Sorted cluster ids for binding site atoms.


    numpy.ndarray
        Sorted cluster ids for all atoms with -1 representing non-site.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None

    bind_coords = all_coords[predicted_labels]
    all_lig_coords = np.row_stack(lig_coord_list)
    ligand_distance = np.min(distance_array(bind_coords, all_lig_coords), axis=1)
    closest_ligand_atom = np.argmin(distance_array(bind_coords, all_lig_coords), axis=1)
    atom_to_ligand_map = np.concatenate([i*np.ones(len(lig_coord_list[i])) for i in range(len(lig_coord_list))])
    cluster_ids = atom_to_ligand_map[closest_ligand_atom]
    cluster_ids[ligand_distance > distance_threshold] = -1
    sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels, score_type=score_type)



    all_ids = -1*np.ones(predicted_labels.shape) 
    all_ids[predicted_labels] = sorted_ids

    return bind_coords, sorted_ids, all_ids

def hull_center(hull):
    """Compute the center of a convex hull.

    Parameters
    ----------
    hull: scipy.spatial.ConvexHull
        Convex hull to compute the center of.

    Returns
    -------
    numpy.ndarray
        Convex hull center of mass.
    """

    hull_com = np.zeros(3)
    tetras = Delaunay(hull.points[hull.vertices])

    for i in range(len(tetras.simplices)):
        tetra_verts = tetras.points[tetras.simplices][i]

        a, b, c, d = tetra_verts
        a, b, c = a - d, b - d, c - d
        tetra_vol = np.abs(np.linalg.det([a, b, c])) / 6

        tetra_com = np.mean(tetra_verts, axis=0)

        hull_com += tetra_com * tetra_vol

    hull_com = hull_com / hull.volume
    return hull_com

def get_centroid(coords):
    """Compute the centroid of a set of coordinates.

    Parameters
    ----------
    coords: numpy.ndarray
        Array of coordinates.

    Returns
    -------
    numpy.ndarray
        Centroid of input coordinates.
    """

    return np.mean(coords, axis=0)

def DCA_dist(center, lig_coords):
    """Compute the distance to the nearest ligand heavy atom for a binding site center.

    Parameters
    ----------
    center: numpy.ndarray
        Binding site center.
    
    lig_coords: numpy.ndarray
        Heavy atom coordinates for all ligands.

    Returns
    -------
    numpy.ndarray
        Minimum distance from site center to ligand heavy atoms.
    """

    distances = np.sqrt(np.sum((center - lig_coords)**2, axis=1))
    shortest = np.min(distances)
    
    return shortest

def hulls_from_clusters(bind_coords, sorted_ids, n_sites):
    """Compute the convex hull for each binding site cluster.

    Parameters
    ----------
    bind_coords: numpy.ndarray
        Coordinates for all binding site atoms.
    
    sorted_ids: numpy.ndarray
        Cluster labels for all binding site atoms.

    n_sites: int
        Maximum number of binding sites.

    Returns
    -------
    list of np.ndarrays
        List of atom coordinates for each site.

    list of scipy.spatial.ConvexHulls
        Convex hull for each site.

    list of np.ndarrays
        Convex hull center for each site.
    """

    top_ids = np.unique(sorted_ids)[::-1][:n_sites]
    predicted_points_list = []
    predicted_hull_list = []
    predicted_center_list = []
    
    for c_id in top_ids:
        if c_id is not None:
            if c_id >= 0:
                predicted_points = bind_coords[sorted_ids == c_id]
                predicted_points_list.append(predicted_points)
                if len(predicted_points) < 4:  # You need four points to define a convex hull
                    predicted_center_list.append(get_centroid(predicted_points))
                    predicted_hull_list.append(None)
                else:
                    predicted_hull = ConvexHull(predicted_points)
                    predicted_hull_list.append(ConvexHull(predicted_points))
                    predicted_center_list.append(hull_center(predicted_hull))

    return predicted_points_list, predicted_hull_list, predicted_center_list

def center_of_probability(bind_coords, bind_probs, sorted_ids, n_sites, type='prob'):
    """Compute the probability-weighted center for each binding site.

    Parameters
    ----------
    bind_coords: numpy.ndarray
        Coordinates for all binding site atoms.

    bind_probs: numpy.ndarray
        Site probabilites for each site atom.
    
    sorted_ids: numpy.ndarray
        Cluster labels for all binding site atoms.

    n_sites: int
        Maximum number of binding sites.

    type: str
        Option for weight type (probability, square of probability,  uniform/centroid).

    Returns
    -------
    list of np.ndarrays
        List of centers for each site.
    """

    top_ids = np.unique(sorted_ids)[::-1][:n_sites]
    predicted_center_list = []
    
    for c_id in top_ids:
        if c_id is not None:
            if c_id >= 0:
                predicted_points = bind_coords[sorted_ids == c_id]
                cluster_probs = bind_probs[:,1][sorted_ids == c_id]
                if type == 'square':
                    cluster_probs = cluster_probs**2
                if type == "centroid":
                    cluster_probs = np.ones(cluster_probs.shape)
                prob_center = center_of_mass(predicted_points, cluster_probs)
                predicted_center_list.append(prob_center)

    return predicted_center_list

def subgraph_adjacency(adjacency, indices):
    """Compute the adjacency matrix for a subgraph induced by a specified index set.

    Parameters
    ----------
    adjacency: numpy.ndarray
        Original adjacency matrix.
    
    indices: numpy.ndarray
        Subset of indices for inducing the subgraph.

    Returns
    -------
    np.ndarray
        Subgraph adjacency matrix.
    """
    
    return adjacency[indices].T[indices].T

def convert_atom_indices_to_array_indices(input_atom_order, atom_array):
    """Convert the indexing from protein atoms to Connolly surface.

    Parameters
    ----------
    adjacency: numpy.ndarray
        Connolly index order.
    
    atom_array: numpy.ndarray
        Original index order.

    Returns
    -------
    np.ndarray
        Map between index orders.
    """
    
    array_indices = np.array([np.where(atom_array == atom)[0][0] for atom in input_atom_order])

    return array_indices

def get_clusters_from_connolly(connolly_vertices, connolly_atoms, tracked_indices, sorted_ids, predicted_probs, threshold):
    """Map atoms and their associated sites to the Connolly surface mesh.

    Parameters
    ----------
    connolly_vertices: numpy.ndarray
        Connolly gridpoint coordinates.

    connolly_atoms: numpy.ndarray
        Atoms corresponding to Connolly gridpoints.
    
    tracked_indices: numpy.ndarray
        Indices of tracked atom set (surface or all).

    sorted_ids: numpy.ndarray
        Sorted bind site cluster for each
    
    predicted_probs: numpy.ndarray
        Tracked atom binding site probabilities.

    threshold: float
        Probability threshold to predict binding site atoms.

    Returns
    -------
    numpy.ndarray
        Binding site mesh coordinates.

    numpy.ndarray
        Mesh cluster membership.

    numpy.ndarray
        Mesh predicted probabilites.
    """

    predicted_labels = predicted_probs[:,1] > threshold
    predicted_probs = predicted_probs[predicted_labels]
    tracked_indices = tracked_indices[predicted_labels]
    
    selected_connolly = np.isin(connolly_atoms, tracked_indices)
    if np.sum(selected_connolly) == 0:
        print('Failed to project atom indices onto connolly.')
        return None, None, None
    connolly_atoms = connolly_atoms[selected_connolly]
    bind_coords = connolly_vertices[selected_connolly]

    where_in_arrays = convert_atom_indices_to_array_indices(connolly_atoms, tracked_indices)
    sorted_ids = sorted_ids[where_in_arrays]
    predicted_probs = predicted_probs[where_in_arrays]


    return bind_coords, sorted_ids, predicted_probs


def scPDB_ligand_merge(lig_coord_list, lig_mass_list, lig_center_list):
    """Merge ligands with center of mass within 5 A as is done in scPDB.

    Parameters
    ----------
    lig_coord_list: list of numpy.ndarrays
        Heavy atom coordinates for each ligand.
    
    lig_mass_list: list of numpy.ndarrays
        Heavy atom masses for each ligand.

    lig_center_list: list of numpy.ndarrays
        Heavy atom center of mass for each ligand.
    
    Returns
    -------
    list of numpy.ndarrays
        Heavy atom coordinates for each ligand after merging.
    
    list of numpy.ndarrays
        Heavy atom masses for each ligand after merging.
    """

    if len(lig_coord_list) == 1:
        return lig_coord_list, lig_center_list
    
    lig_adjacency = contact_matrix(np.array(lig_center_list), cutoff=5)
    if np.sum(lig_adjacency) == len(lig_coord_list): 
        return lig_coord_list, lig_center_list
    
    new_coords = []
    new_centers = []
    lig_connections = nx.connected_components(nx.from_numpy_matrix(lig_adjacency))
    for merge in lig_connections:
        merge_coords = [lig_coord_list[i] for i in merge]
        if len(merge) == 1:
            new_coords += merge_coords
            new_centers += [lig_center_list[i] for i in merge]
        else:
            merge_coords = np.row_stack(merge_coords)
            merge_masses = np.concatenate([lig_mass_list[i] for i in merge])
            merge_centers = center_of_mass(merge_coords, merge_masses)
            new_coords.append(merge_coords)
            new_centers.append(merge_centers)
    
    return new_coords, new_centers


def multisite_metrics(prot_coords, lig_coord_list, lig_mass_list, predicted_probs, top_n_plus=0, 
threshold=.5, eps=3, resolution=0.05, method="louvain", score_type="mean", centroid_type="hull", 
cluster_all=False, adj_matrix=None, surf_mask=None, connolly_data=None, tracked_indices=None, 
ligand_merge=False, known_n_sites=True):
    """Cluster multiple binding sites and calculate distance from each ligand.

    Parameters
    ----------
    prot_coords: numpy.ndarray
        Protein atomic coordinates.
        
    lig_coord_list: list of numpy.ndarrays
        Ligand atomic coordinates for each ligand.
        
    lig_mass_list: list of numpy.ndarrays
        Ligand atomic masses for each ligand.

    predicted_probs: list of numpy.ndarrays
        Class probabilities for not site in column 0 and site in column 1.
        
    top_n_plus: int
        Number of predicted sites to include compared to number of true sites (eg. 1 means 1 more predicted).

    threshold: float
        Probability threshold to predict binding site atoms.

    eps: float
        Distance threshold used for clustering.

    resolution: float
        Resolution only used for Louvain communuty detection.

    method: str
        Clustering method to use.

    score_type: str
        Score function to aggregate atomic scores into site scores.

    centroid_type: str
        Function to compute the site centers.

    cluster_all: bool
        Whether to assign points outside kernels to the nearest cluster or leave them unlabeled.

    adj_matrix: numpy.ndarray
        Adjacency for Louvain community detection.

    surf_mask: numpy.ndarray
        Mask for selecting only surface atoms.

    connolly_data: dict
        Connolly surface points for optional surface mesh clustering.

    tracked_indices: numpy.ndarray
        Indices for optional Connolly surface clustering.

    ligand_merge: bool
        Whether to merge ligands with center of mass within 5 A as is done in scPDB.

    known_n_sites: bool
        Whether to use the ground truth number of sites (True) or a static maximum (False).

    Returns
    -------
    numpy.ndarray
        Array of distances from predicted site center to ligand center of mass. 
        
    numpy.ndarray
        Array of closest distances from predicted site center to any ligand heavy atom. 

    int
        Total number of binding sites predicted.

    int 
        Number of sites predicted within the limit (top N+M or static M).
    """

    if surf_mask is not None:
        prot_coords = prot_coords[surf_mask]
        if method == "louvain":
            adj_matrix = subgraph_adjacency(adj_matrix, surf_mask)
    if connolly_data is not None:
        connolly_vertices = connolly_data['vertices']
        connolly_atoms = connolly_data['atom_indices']
        tracked_indices = tracked_indices[surf_mask]

    if method == "meanshift":
        bind_coords, sorted_ids, all_ids = cluster_atoms_meanshift(prot_coords, predicted_probs, threshold=threshold, cluster_all=cluster_all, score_type=score_type)
    elif method == "dbscan":
        bind_coords, sorted_ids, all_ids = cluster_atoms_DBSCAN(prot_coords, predicted_probs, threshold=threshold, eps=eps, score_type=score_type)
    elif method == "louvain":
        bind_coords, sorted_ids, all_ids = cluster_atoms_louvain(prot_coords,adj_matrix,predicted_probs,threshold=threshold, cutoff=eps, resolution=resolution, score_type=score_type)
    elif method == "single":
        bind_coords, sorted_ids, all_ids = cluster_atoms_single(prot_coords, predicted_probs, threshold=threshold, n_clusters=None, distance_threshold=eps, score_type=score_type)
    elif method == "complete":
        bind_coords, sorted_ids, all_ids = cluster_atoms_complete(prot_coords, predicted_probs, threshold=threshold, n_clusters=None, distance_threshold=eps, score_type=score_type)
    elif method == "average":
        bind_coords, sorted_ids, all_ids = cluster_atoms_average(prot_coords, predicted_probs, threshold=threshold, n_clusters=None, distance_threshold=eps, score_type=score_type)
    elif method == "ward":
        bind_coords, sorted_ids, all_ids = cluster_atoms_ward(prot_coords, predicted_probs, threshold=threshold, n_clusters=None, distance_threshold=eps, score_type=score_type)
    elif method == "groundtruth":
        bind_coords, sorted_ids, all_ids = cluster_atoms_groundtruth(prot_coords, lig_coord_list, predicted_probs, threshold=threshold, distance_threshold=eps, score_type=score_type)

    if connolly_data is not None:
        bind_coords, sorted_ids, predicted_probs = get_clusters_from_connolly(connolly_vertices, connolly_atoms, tracked_indices, sorted_ids, predicted_probs, threshold)
        
    lig_center_list = [center_of_mass(lig_coord_list[i], lig_mass_list[i]) for i in range(len(lig_coord_list))]

    if ligand_merge:
        lig_coord_list, lig_center_list = scPDB_ligand_merge(lig_coord_list, lig_mass_list, lig_center_list)

    if known_n_sites: n_sites = len(lig_center_list)
    else: n_sites = 0 # just use the static term

    if centroid_type == "hull":
        _, _, predicted_center_list = hulls_from_clusters(bind_coords, sorted_ids, n_sites+top_n_plus)
    else:
        bind_probs = predicted_probs[predicted_probs[:,1] > threshold]
        predicted_center_list = center_of_probability(bind_coords, bind_probs, sorted_ids, n_sites+top_n_plus, type=centroid_type)

    if type(sorted_ids) == type(None):
        n_predicted = 0
        top_predicted = 0
    else:
        n_predicted = len(np.unique(sorted_ids[sorted_ids >= 0]))
        top_predicted = len(predicted_center_list)

    if len(predicted_center_list) > 0:          
        DCC_lig_matrix = np.zeros([len(lig_center_list), len(predicted_center_list)])
        DCA_matrix = np.zeros([len(lig_center_list), len(predicted_center_list)])

        for index, x in np.ndenumerate(DCA_matrix):
            true_ind, pred_ind = index

            predicted_center = predicted_center_list[pred_ind]
            lig_center = lig_center_list[true_ind]
            lig_coords = lig_coord_list[true_ind]

            DCC_lig_matrix[index] = np.sqrt(np.sum((predicted_center - lig_center)**2))
            DCA_matrix[index] = DCA_dist(predicted_center, lig_coords)

        DCC_lig = np.min(DCC_lig_matrix, axis=1)
        DCA = np.min(DCA_matrix, axis=1)

        return DCC_lig, DCA, n_predicted, top_predicted

    else:
        nan_arr =  np.empty(len(lig_center_list))
        nan_arr[:] = np.nan

        return nan_arr, nan_arr, n_predicted, top_predicted


def compute_metrics_for_all(path_to_mol2, path_to_labels, top_n_plus=0, threshold = 0.5, eps=3, resolution=0.05,
method="louvain", score_type="mean", centroid_type="hull", cluster_all=False, use_surface=False, use_connolly=False, 
ligand_merge=False, known_n_sites=True):
    """Cluster multiple binding sites and calculate distance from each ligand.

    Parameters
    ----------
    path_to_mol2: str
        Path to protein mol2 files.

    path_to_labels: str
        Path to atomic class (site/non-site) labels.
        
    top_n_plus: int
        Number of predicted sites to include compared to number of true sites (eg. 1 means 1 more predicted).

    threshold: float
        Probability threshold to predict binding site atoms.

    eps: float
        Distance threshold used for clustering.

    resolution: float
        Resolution only used for Louvain communuty detection.

    method: str
        Clustering method to use.

    score_type: str
        Score function to aggregate atomic scores into site scores.

    centroid_type: str
        Function to compute the site centers.

    cluster_all: bool
        Whether to assign points outside kernels to the nearest cluster or leave them unlabeled.

    use_surface: bool
        Whether to only use surface atoms.

    use_connolly: bool
        Whether to use a Connolly surface mesh.

    ligand_merge: bool
        Whether to merge ligands with center of mass within 5 A as is done in scPDB.

    known_n_sites: bool
        Whether to use the ground truth number of sites (True) or a static maximum (False).

    Returns
    -------
    list of numpy.ndarrays
        List of DCC for each system. 
        
    list of numpy.ndarrays
        List of DCA for each system.

    list of int
        List of total number of binding sites predicted for each system.

    list of int 
        List of number of sites predicted within the limit (top N+M or static M) for each system.

    list of int
        Number of systems without predictions.

    list of str
        System names.
    """

    DCC_lig_list = []
    DCA_list = []

    def helper(file):
        """Helper function to load and analyze each system.

        Parameters
        ----------
        file: str
            System file path.

        Returns
        -------
        numpy.ndarray
            Array of distances from predicted site center to ligand center of mass. 
            
        numpy.ndarray
            Array of closest distances from predicted site center to any ligand heavy atom. 

        int
            Total number of binding sites predicted.

        int 
            Number of sites predicted within the limit (top N+M or static M).

        int
            Number of systems without predictions.
        """

        no_prediction_count = 0
        assembly_name = file.split('.')[-2]
        try:
            trimmed_protein = mda.Universe(path_to_mol2 + assembly_name + '.mol2')
            labels = np.load(prepend + metric_dir + '/labels/' + model_name + '/' + assembly_name + '.npy')
            probs = np.load(prepend + metric_dir + '/probs/' + model_name + '/' + assembly_name + '.npy')
            atom_indices = np.load(prepend + metric_dir + '/indices/' + model_name + '/' + assembly_name + '.npy')

            surf_mask = None
            connolly_data = None
            tracked_indices = None
            if use_surface: 
                surf_mask = np.load(prepend + metric_dir + '/SASAs/'  + assembly_name + '.npy')
            if use_connolly:
                connolly_dir = '/'.join(path_to_mol2.split('/')[:-2])+'/connolly'
                connolly_data = np.load(f'{connolly_dir}/{assembly_name}.npz')
                tracked_indices = atom_indices

            if is_label: probs = labels

            lig_coord_list = []
            lig_mass_list = []
            
            for file_path in sorted(glob(data_dir + '/ready_to_parse_mol2/' + assembly_name + '/*')):
                if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
                    ligand = mda.Universe(file_path)
                    lig_coord_list.append(list(ligand.atoms.positions))
                    lig_mass_list.append(list(ligand.atoms.masses))
            adj_matrix = np.load(data_dir+'/raw/' + assembly_name + '.npz', allow_pickle=True)['adj_matrix'].item()
            adj_matrix = subgraph_adjacency(adj_matrix, atom_indices)
            DCC_lig, DCA, n_predicted, top_predicted = multisite_metrics(trimmed_protein.atoms[atom_indices].positions, lig_coord_list, lig_mass_list,
                probs, top_n_plus=top_n_plus, threshold=threshold, eps=eps, resolution=resolution, method=method, score_type=score_type,
                centroid_type=centroid_type, cluster_all=cluster_all, adj_matrix=adj_matrix, surf_mask=surf_mask, connolly_data=connolly_data,
                tracked_indices=tracked_indices, ligand_merge=ligand_merge, known_n_sites=known_n_sites)

            if np.all(np.isnan(DCC_lig)) and np.all(np.isnan(DCA)): 
                no_prediction_count += 1
            return DCC_lig, DCA, n_predicted, top_predicted, no_prediction_count

        except Exception as e:
            print("ERROR")
            print(assembly_name, flush=True)
            raise e

    r = Parallel(n_jobs=n_jobs)(delayed(helper)(file) for file in tqdm(os.listdir(path_to_labels)[:],  position=0, leave=True))
    DCC_lig_list, DCA_list, n_predicted, top_predicted, no_prediction_count = zip(*r)
    names = [file for file in os.listdir(path_to_labels)]
    return DCC_lig_list, DCA_list, n_predicted, top_predicted, no_prediction_count, names

def criteria_to_metrics(metric_array, top_predicted):
    """Compute the precision and recall for a given criteria.

    Parameters
    ----------
    metric_array: list of numpy.ndarray
        Distance measure (DCC or DCA) for each ligand in each system.
    
    top_predicted: list of int
        Number of predictions subject to maximum number of prediction (top N+M or M) for each system.

    Returns
    -------
    float
        Recall on the input criteria.

    float
        Precison on the input criteria.
    """

    recall = np.mean(np.concatenate(metric_array) <= 4)

    # for precision we don't want to count a success if it exceeds the number of predictions
    # (from prediction site being within 4 A of multiple sites)
    capped_success = [min(np.sum(metric_array[i] <= 4), top_predicted[i]) for i in range(len(metric_array))]
    precision = np.sum(capped_success) / np.sum(top_predicted)
        
    return recall, precision

#######################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster GNN predictions into binding sites.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("test_set", choices=["val", "coach420", "coach420_mlig", "coach420_intersect",
     "holo4k", "holo4k_mlig", "holo4k_intersect", "holo4k_chains"], help="Test set.")
    parser.add_argument("model_name", help="Model file path.")
    parser.add_argument("-c", "--clustering_method", default="average", choices=["meanshift", "dbscan", "louvain", "single", "complete", "average", "ward", "groundtruth"], help="Clustering method.")
    parser.add_argument("-d", "--dist_thresholds", type=float, nargs="+", default=[15], help="Distance thresholds for clustering.")
    parser.add_argument("-p", "--prob_threshold", type=float, default=.3, help="Probability threshold for atom classification.")
    parser.add_argument("-np", "--top_n_plus", type=int, nargs="+", default=[0,2,1000], help="Number of additional sites to consider.")
    parser.add_argument("-tn", "--top_n", type=int, nargs="+", default=[3,5], help="Static number of sites to consider.")
    parser.add_argument("-o", "--compute_optimal", action="store_true", help="Option to compute optimal threshold.")
    parser.add_argument("-l", "--use_labels", action="store_true", help="Option to cluster true labels.")
    parser.add_argument("-ao", "--all_atom_prediction", action="store_true", help="Option to perform inference on all atoms as opposed to solvent exposed.")
    parser.add_argument("-uc", "--use_connolly", action="store_true", help="Project clusters onto Connolly surface to make sites.")
    parser.add_argument("-a", "--aggregation_function", default="square", choices=["mean", "sum", "square"], help="Function to combine atom scores into site scores.")
    parser.add_argument("-r", "--louvain_resolution", type=float, default=0.05, help="Resolution for Louvain community detection (not used in other methods).")
    parser.add_argument("-ct", "--centroid_type", default="hull", choices=["hull", "prob", "square", "centroid"], help="Type of centroid to use for site center.")
    parser.add_argument("-lm", "--ligand_merge", action="store_true", help="Merge ligands that have center of mass within 5 A like in scPDB.")
    parser.add_argument("-n", "--n_tasks", type=int, default=15, help="Number of cpu workers.")

    args = parser.parse_args()
    non_path_args = [sys.argv[1]] + sys.argv[3:]
    argstring='_'.join(non_path_args).replace('-','')

    model_name = args.model_name

    prepend = str(os.getcwd())
    eps_list = args.dist_thresholds
    threshold = args.prob_threshold
    resolution = args.louvain_resolution
    method = args.clustering_method
    compute_optimal = args.compute_optimal
    top_n_list = args.top_n_plus
    top_list = args.top_n
    score_type = args.aggregation_function
    centroid_type = args.centroid_type
    use_surface = not args.all_atom_prediction
    use_connolly = args.use_connolly
    n_jobs = args.n_tasks
    ligand_merge = args.ligand_merge

    is_label=args.use_labels
    if is_label:
        print("Using labels rather than probabilities.")
    if use_surface:
        print("Using surface atoms to predict sites.")
    if use_connolly:
        print("Projecting sites onto Connolly surface.")
    if ligand_merge:
        print("scPDB style ligand merging.")

    set_to_use = args.test_set
    if set_to_use == 'val':
        print("Calculating metrics on the validation set")
        data_dir = prepend + '/scPDB_data_dir'
        metric_dir = '/test_metrics/validation'
    else:
        print(f"Calculating metrics on the {set_to_use} set")    
        data_dir = f'{prepend}/benchmark_data_dir/{set_to_use}'
        metric_dir = f'/test_metrics/{set_to_use}'

    #######################################################################################
    if compute_optimal:
        all_prob_path = prepend + metric_dir + '/all_probs/' + model_name + '/'
        all_label_path = prepend + metric_dir + '/all_labels/' + model_name + '/'
        all_probs  = np.load(all_prob_path + "all_probs.npz")['arr_0']
        all_labels = np.load(all_label_path + "all_labels.npz")['arr_0']
        start = time.time()

        binarized_labels = np.array([[0,1] if x == 1 else [1,0] for x in all_labels])
        # Compute roc, auc and optimal threshold
        all_probs = np.array(all_probs, dtype=object)

        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        n_classes = 2

        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(binarized_labels[:, i],all_probs[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(binarized_labels.ravel(), all_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_path = prepend + metric_dir + '/roc_curves/' + model_name

        if not os.path.isdir(roc_path):
            os.makedirs(roc_path)

        # Find optimal threshold
        gmeans = np.sqrt(tpr[1] * (1-fpr[1]))
        ix = np.argmax(gmeans)
        optimal_threshold = thresholds[1][ix]
        threshold_lst.insert(0, optimal_threshold)

        print('Best Threshold=%f, G-Mean=%.3f' % (optimal_threshold, gmeans[ix]))
        print("Micro Averaged AUC:", roc_auc["micro"])
        print("Macro Averaged AUC:", roc_auc["macro"])
        print("Negative Class AUC:", roc_auc[0])
        print("Positive Class AUC:", roc_auc[1])

        print("Done. {}".format(time.time()- start))
        
    #######################################################################################



    #######################################################################################
    outdir = f"{prepend}{metric_dir}/clustering/{model_name}"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = f"{outdir}/{argstring}.dat"
    if os.path.exists(outfile):
        os.remove(outfile)
    out = open(outfile, 'a')
    for eps in eps_list:
        for top_n_plus in top_n_list:
            print(f"Calculating n+{top_n_plus} metrics for {threshold} threshold with distance cutoff {eps}.", flush=True)
            out.write(f"Calculating n+{top_n_plus} metrics for {threshold} threshold with distance cutoff {eps}.\n")
            start = time.time()
            path_to_mol2= data_dir + '/mol2/'
            path_to_labels= prepend + metric_dir + '/labels/' + model_name + '/'
            DCC_lig, DCA, n_predicted, top_predicted, no_prediction_count, names = compute_metrics_for_all(
            path_to_mol2, path_to_labels, top_n_plus=top_n_plus, threshold=threshold, eps=eps, resolution=resolution,
            method=method, score_type=score_type, centroid_type=centroid_type, use_surface=use_surface, use_connolly=use_connolly, ligand_merge=ligand_merge)

            print("Done. {}".format(time.time()- start))
            out.write("Done. {}\n".format(time.time()- start))
            
            overlap_path = f"{prepend}{metric_dir}/overlaps/{model_name}"
            if not os.path.isdir(overlap_path):
                os.makedirs(overlap_path)
            

            np.savez(f"{overlap_path}/{argstring}_n+{top_n_plus}.npz", DCC_lig=np.array(DCC_lig, dtype=object), DCA=np.array(DCA, dtype=object),
             n_predicted=n_predicted, names=names)

            n_predicted = np.array(n_predicted)

            print("-----------------------------------------------------------------------------------", flush=True)
            print(f"Method: {method}")
            print(f"Cutoff (Prediction Threshold): {threshold}")
            print(f"EPS: {eps}")
            print(f"top n + {top_n_plus} prediction")
            print("-----------------------------------------------------------------------------------", flush=True)
            print(f"Number of systems with no predictions: {np.sum(no_prediction_count)}", flush=True)

            out.write(f"Method: {method}\n")
            out.write("-----------------------------------------------------------------------------------\n")
            out.write(f"Cutoff (Prediction Threshold): {threshold}\n")
            out.write(f"EPS: {eps}\n")
            out.write(f"top n + {top_n_plus} prediction\n")
            out.write("-----------------------------------------------------------------------------------\n")
            out.write(f"Number of systems with no predictions: {np.sum(no_prediction_count)}\n")
    
            DCC_lig_recall, DCC_lig_precision = criteria_to_metrics(DCC_lig, top_predicted)
            DCA_recall, DCA_precision = criteria_to_metrics(DCA, top_predicted)

            
            print(f"DCC_lig Recall: {DCC_lig_recall}", flush=True)
            print(f"DCC_lig Precision: {DCC_lig_precision}", flush=True)

            print(f"DCA Recall: {DCA_recall}", flush=True)
            print(f"DCA Precision: {DCA_precision}", flush=True)

            print(f"Average n_predicted: {np.nanmean(n_predicted)}", flush=True)

            
            out.write(f"DCC_lig Recall: {DCC_lig_recall}\n")
            out.write(f"DCC_lig Precision: {DCC_lig_precision}\n")

            out.write(f"DCA Recall: {DCA_recall}\n")
            out.write(f"DCA Precision: {DCA_precision}\n")

            out.write(f"Average n_predicted: {np.nanmean(n_predicted)}\n")
            #######################################################################################

        for top in top_list:
            print(f"Calculating top {top} metrics for {threshold} threshold with distance cutoff {eps}.", flush=True)
            out.write(f"Calculating top {top} metrics for {threshold} threshold with distance cutoff {eps}.\n")
            start = time.time()
            path_to_mol2= data_dir + '/mol2/'
            path_to_labels= prepend + metric_dir + '/labels/' + model_name + '/'
            DCC_lig, DCA, n_predicted, top_predicted, no_prediction_count, names = compute_metrics_for_all(
            path_to_mol2, path_to_labels, top_n_plus=top, threshold=threshold, eps=eps, resolution=resolution,
            method=method, score_type=score_type, centroid_type=centroid_type, use_surface=use_surface, use_connolly=use_connolly, 
            ligand_merge=ligand_merge, known_n_sites=False)

            print("Done. {}".format(time.time()- start))
            out.write("Done. {}\n".format(time.time()- start))
            
            overlap_path = f"{prepend}{metric_dir}/overlaps/{model_name}"
            if not os.path.isdir(overlap_path):
                os.makedirs(overlap_path)
            

            np.savez(f"{overlap_path}/{argstring}_top{top}.npz", DCC_lig=np.array(DCC_lig, dtype=object), DCA=np.array(DCA, dtype=object),
             n_predicted=n_predicted, names=names)

            n_predicted = np.array(n_predicted)

            print("-----------------------------------------------------------------------------------", flush=True)
            print(f"Method: {method}")
            print(f"Cutoff (Prediction Threshold): {threshold}")
            print(f"EPS: {eps}")
            print(f"top {top} prediction")
            print("-----------------------------------------------------------------------------------", flush=True)
            print(f"Number of systems with no predictions: {np.sum(no_prediction_count)}", flush=True)

            out.write(f"Method: {method}\n")
            out.write("-----------------------------------------------------------------------------------\n")
            out.write(f"Cutoff (Prediction Threshold): {threshold}\n")
            out.write(f"EPS: {eps}\n")
            out.write(f"top {top} prediction\n")
            out.write("-----------------------------------------------------------------------------------\n")
            out.write(f"Number of systems with no predictions: {np.sum(no_prediction_count)}\n")
    
            DCC_lig_recall, DCC_lig_precision = criteria_to_metrics(DCC_lig, top_predicted)
            DCA_recall, DCA_precision = criteria_to_metrics(DCA, top_predicted)

            
            print(f"DCC_lig Recall: {DCC_lig_recall}", flush=True)
            print(f"DCC_lig Precision: {DCC_lig_precision}", flush=True)

            print(f"DCA Recall: {DCA_recall}", flush=True)
            print(f"DCA Precision: {DCA_precision}", flush=True)

            print(f"Average n_predicted: {np.nanmean(n_predicted)}", flush=True)

            
            out.write(f"DCC_lig Recall: {DCC_lig_recall}\n")
            out.write(f"DCC_lig Precision: {DCC_lig_precision}\n")

            out.write(f"DCA Recall: {DCA_recall}\n")
            out.write(f"DCA Precision: {DCA_precision}\n")

            out.write(f"Average n_predicted: {np.nanmean(n_predicted)}\n")
            #######################################################################################
    out.close()
