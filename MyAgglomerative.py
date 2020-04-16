import time
import numpy as np 
import sys
#from scipy.spatial import distance 
#from scipy.cluster import hierarchy
from heapq import heappush, heappushpop
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import load_iris

def find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,cluster_2, linkage):
    if (cluster_tree_dict[cluster_1][1]==-1 and cluster_tree_dict[cluster_2][1]==-1):
        return distance_matrix[cluster_1][cluster_2]
    
    elif (cluster_tree_dict[cluster_1][1]!=-1):
        child_1 = cluster_tree_dict[cluster_1][0]
        child_2 = cluster_tree_dict[cluster_1][1]

        if (linkage=="single"):
            return min(find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_1,cluster_2,linkage),
            find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_2,cluster_2,linkage))
        elif (linkage=="complete"):
            return max(find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_1,cluster_2,linkage),
            find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_2,cluster_2,linkage))
        elif (linkage=="average"):
            return find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_1,cluster_2,linkage) + find_distance_between_cluster(distance_matrix,cluster_tree_dict,child_2,cluster_2,linkage)
            
    
    elif (cluster_tree_dict[cluster_2][1]!=-1):
        child_1 = cluster_tree_dict[cluster_2][0]
        child_2 = cluster_tree_dict[cluster_2][1]

        if (linkage=="single"):
            return min(find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_1,linkage),
            find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_2,linkage))
        elif (linkage=="complete"):
            return max(find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_1,linkage),
            find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_2,linkage))
        elif (linkage=="average"):
            return find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_1,linkage) + find_distance_between_cluster(distance_matrix,cluster_tree_dict,cluster_1,child_2,linkage)
            

def hc_cluster(X,metric='euclidean',linkage='single'):
    
    cluster_tree_dict = dict(list(enumerate([[i,-1] for i in range(len(X))])))
    
    cluster_group = dict(list(enumerate([1 for i in range(len(X))])))
    
    initial_distances = pairwise_distances(X,metric=metric)
    #print(initial_distances)    
    np.fill_diagonal(initial_distances,sys.maxsize)

    distance_matrix = dict(list(enumerate([initial_distances[i,j] for i in range(initial_distances.shape[0])] for j in range(initial_distances.shape[1]))))
    #distance_matrix = dict(list(enumerate( initial_distances[j] for j in range(initial_distances.shape[1]))))
    #print(distance_matrix)
    
    cluster_tree = np.zeros((len(X)-1,4))
    for i in range(len(X)-1):
        cluster_1 = cluster_2 = -1
        cluster_val = 0
        min_dist = sys.maxsize
        #print("me")
        #find min distance of 2 cluster
        if (linkage=="single" or linkage=="complete"):
            for c1 in cluster_group:
                for c2 in cluster_group:
                    if (c1<c2):
                        #print(str(c1)+ " " + str(c2))
                        dist = find_distance_between_cluster(distance_matrix,cluster_tree_dict,c1,c2,linkage)
                        if (dist<min_dist):
                            min_dist = dist
                            cluster_1 = c1
                            cluster_2 = c2
                            cluster_val = cluster_group[c1]+cluster_group[c2]
                        last_cluster = c2
        elif (linkage=="average"):
            for c1 in cluster_group:
                for c2 in cluster_group:
                    if (c1<c2):
                        #print(str(c1)+ " " + str(c2))
                        dist = find_distance_between_cluster(distance_matrix,cluster_tree_dict,c1,c2,linkage)/(cluster_group[c1]*cluster_group[c2])
                        if (dist<min_dist):
                            min_dist = dist
                            cluster_1 = c1
                            cluster_2 = c2
                            cluster_val = cluster_group[c1]+cluster_group[c2]
                        last_cluster = c2
        #print(last_cluster)

        cluster_tree_dict[last_cluster+1] = [cluster_1, cluster_2]
        cluster_group[last_cluster+1] = cluster_val
        #print
        cluster_tree[i][0] = cluster_1
        cluster_tree[i][1] = cluster_2
        cluster_tree[i][2] = min_dist
        cluster_tree[i][3] = cluster_group[last_cluster+1]
        #print(min_dist)
        #print(cluster_group)
        del cluster_group[cluster_1]
        del cluster_group[cluster_2]
    
    return cluster_tree
    #print(cluster_tree)

#belum di implementasi
def ward_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    X = np.require(X, requirements="W")

    dm = distance.pdist(X,'euclidean')

    out = hc_cluster(X,linkage="ward")
    #out = hierarchy.ward(X)
    #print(out)
    children_ = out[:, :2].astype(np.intp)
    return children_, 1, n_samples
    
def complete_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    X = np.require(X, requirements="W")

    #out = hierarchy.complete(X)
    out = hc_cluster(X,linkage="complete")
    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

def average_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    X = np.require(X, requirements="W")

    #out = hierarchy.average(X)
    out = hc_cluster(X,linkage="average")
    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

def single_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    X = np.require(X, requirements="W")
    #out = hierarchy.single(X)
    out = hc_cluster(X,linkage="single")
    #print(out)

    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

TREE_BUILDER = dict(
    ward=ward_tree,
    complete=complete_tree,
    average=average_tree,
    single=single_tree)

def hc_get_descendent(node, children,n_leaves):
    """
    Function returning all the descendent leaves of a set of nodes in the tree.
    Parameters
    ----------
    node : integer
        The node for which we want the descendents.
    children : list of pairs, length n_nodes
        The children of each non-leaf node. Values less than `n_samples` refer
        to leaves of the tree. A greater value `i` indicates a node with
        children `children[i - n_samples]`.
    n_leaves : integer
        Number of leaves.
    Returns
    -------
    descendent : list of int
    """
    ind = [node]
    if node < n_leaves:
        return ind
    descendent = []

    # It is actually faster to do the accounting of the number of
    # elements is the list ourselves: len is a lengthy operation on a
    # chained list
    n_indices = 1

    while n_indices:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
            n_indices -= 1
        else:
            ind.extend(children[i - n_leaves])
            n_indices += 1
    return descendent

def cut_tree(n_clusters, children, n_leaves):
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    
    nodes = [-(max(children[-1]) + 1)]
    #print(nodes)

    for i in range(n_clusters -1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    #print(nodes)
    label = np.zeros(n_leaves, dtype=np.intp)

    for i,node in enumerate(nodes):
        label[hc_get_descendent(-node,children,n_leaves)] = i
    return label

class MyAgglomerative():
    def __init__(self,n_clusters=2,affinity='euclidean',linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity

    def fit(self,X):
        n_samples = len(X)
        n_clusters = self.n_clusters

        tree_builder = TREE_BUILDER[self.linkage]
        
        self.children_, self.n_connected_components_, self.n_leaves_ = tree_builder(X,n_clusters=n_clusters)
        
        self.lables_ = cut_tree(n_clusters, self.children_, self.n_leaves_)

        return self

    def predict(self,x):
        pass


#test
if __name__ == '__main__':
    """
    X = np.array([[5,3],
        [10,15],
        [15,12],
        [24,10],
        [30,30],
        [85,70],
        [71,80],
        [60,78],
        [70,55],
        [80,91],])
    """
    dataset = load_iris()
    X = dataset.data

    cluster = MyAgglomerative(n_clusters=3,affinity='euclidean',linkage='average')
    tic = time.perf_counter()
    cluster.fit(X)
    toc = time.perf_counter()
    print(f"total myagglomerative process take {toc - tic:0.4f} seconds")
    
    print(cluster.lables_)
    #d = MyAgglo(n_clusters=3,linkage='complete')
    #d.fit(X)
    #print(d.cluster_to_plot)
    #tic = time.perf_counter()
    #out = hc_cluster(X,linkage='average')
    #toc = time.perf_counter()
    #print(f"mine process take {toc - tic:0.4f} seconds")
    #print(out)
    #a = np.zeros((len(X),len(X)))
    #np.put(a,,1 )