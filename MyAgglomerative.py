import time
import numpy as np 
import sys
#from scipy.cluster import hierarchy
from heapq import heappush, heappushpop
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import load_iris

def cluster_mean(X,cluster_tree_dict,cluster):
    if (cluster_tree_dict[cluster][1]==-1):
        return X[cluster_tree_dict[cluster][0]]
    else:
        child_1 = cluster_tree_dict[cluster][0]
        child_2 = cluster_tree_dict[cluster][1]

        return np.vstack((cluster_mean(X,cluster_tree_dict,child_1),cluster_mean(X,cluster_tree_dict,child_2)))

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
    
    if (linkage=="complete" or linkage=="single" or linkage=="average"):
        initial_distances = pairwise_distances(X,metric=metric)
        distance_matrix = dict(list(enumerate([initial_distances[i,j] for i in range(initial_distances.shape[0])] for j in range(initial_distances.shape[1]))))
        cluster_group = dict(list(enumerate([1 for i in range(len(X))])))
       
    elif (linkage=="centroid"):
        cluster_group = dict(list(enumerate(X[i] for i in range(len(X)))))
        
    
    cluster_tree = np.zeros((len(X)-1,4))
    for i in range(len(X)-1):
        min_dist = sys.maxsize

        #find min distance of 2 cluster
        if (linkage=="single" or linkage=="complete"):
            for c1 in cluster_group:
                for c2 in cluster_group:
                    if (c1<c2):
                        dist = find_distance_between_cluster(distance_matrix,cluster_tree_dict,c1,c2,linkage)
                        if (dist<min_dist):
                            min_dist = dist
                            cluster_1 = c1
                            cluster_2 = c2

                        last_cluster = c2

        elif (linkage=="average"):
            for c1 in cluster_group:
                for c2 in cluster_group:
                    if (c1<c2):
                        dist = find_distance_between_cluster(distance_matrix,cluster_tree_dict,c1,c2,linkage)/(cluster_group[c1]*cluster_group[c2])
                        if (dist<min_dist):
                            min_dist = dist
                            cluster_1 = c1
                            cluster_2 = c2

                        last_cluster = c2
                        
        elif (linkage=="centroid"):
            for c1 in cluster_group:
                c1_means = cluster_group[c1]

                for c2 in cluster_group:
                    if (c1<c2):
                        c2_means = cluster_group[c2]
                        dist = np.sqrt(np.sum((c2_means-c1_means)**2))
                       
                        if (dist<min_dist):
                            min_dist = dist
                            cluster_1 = c1
                            cluster_2 = c2
                        last_cluster = c2
        
        cluster_tree_dict[last_cluster+1] = [cluster_1, cluster_2]

        if (linkage=="centroid"):
            c2_data = cluster_mean(X,cluster_tree_dict,last_cluster+1)
            cluster_group[last_cluster+1] = np.mean(c2_data,axis=0)
            cluster_tree[i][3] = last_cluster+1
            
        elif (linkage=="single" or linkage=="complete" or linkage=="average"):
            cluster_group[last_cluster+1] = cluster_group[cluster_1] + cluster_group[cluster_2]
            cluster_tree[i][3] = cluster_group[last_cluster+1]

        cluster_tree[i][0] = cluster_1
        cluster_tree[i][1] = cluster_2
        cluster_tree[i][2] = min_dist

        del cluster_group[cluster_1]
        del cluster_group[cluster_2]
    
    return cluster_tree

#belum di implementasi
def centroid_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    #out = hierarchy.centroid(X)
    out = hc_cluster(X,linkage="centroid")
    #print(out)
    children_ = out[:, :2].astype(np.intp)
    return children_, 1, n_samples
    
def complete_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    #out = hierarchy.complete(X)
    out = hc_cluster(X,linkage="complete")
    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

def average_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    #out = hierarchy.average(X)
    out = hc_cluster(X,linkage="average")
    #print(out)
    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

def single_tree(X,n_clusters=2):
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (-1,1))
    n_samples, n_features = X.shape

    #out = hierarchy.single(X)
    out = hc_cluster(X,linkage="single")
    #print(out)

    children_ = out[:, :2].astype(np.intp)

    return children_, 1, n_samples

TREE_BUILDER = dict(
    centroid=centroid_tree,
    complete=complete_tree,
    average=average_tree,
    single=single_tree)

def get_descendent(node, children,n_leaves):
    
    ind = [node]
    if node < n_leaves:
        return ind
    descendent = []

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

    for i in range(n_clusters -1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    #print(nodes)
    label = np.zeros(n_leaves, dtype=np.intp)

    for i,node in enumerate(nodes):
        label[get_descendent(-node,children,n_leaves)] = i
    return label

class MyAgglomerative():
    def __init__(self,n_clusters=2,affinity='euclidean',linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity

    def fit(self,X):
        tree_builder = TREE_BUILDER[self.linkage]
        
        self.children_, self.n_connected_components_, self.n_leaves_ = tree_builder(X,n_clusters=self.n_clusters)
        
        self.lables_ = cut_tree(self.n_clusters, self.children_, self.n_leaves_)


    def fit_predict(self,X):
        self.fit(X)

        return self.lables_


#test
#"""
if __name__ == '__main__':
    
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
    
    dataset = load_iris()
    #X = dataset.data
    
    cluster = MyAgglomerative(n_clusters=3,affinity='euclidean',linkage='average')
    tic = time.perf_counter()
    cluster.fit(X)
    toc = time.perf_counter()
    print(f"total myagglomerative process take {toc - tic:0.4f} seconds")
    
    print(cluster.lables_)
#"""