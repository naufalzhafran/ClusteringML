import numpy as np 


class MyKMeans():
    def __init__(self,K):
        self.K = K

    def fit(self,x):
        self.centroids = x[np.random.choice(len(x), self.K, replace=False)]
        self.prev_class = None
        self.classes = np.zeros(len(x))
        while not np.all(self.classes == self.prev_class) :
            self.prev_class = self.classes
            self.classes = self.predict(x)
            self.update_centroid(x)

    def predict(self, x):
        return np.apply_along_axis(self.compute_class, 1, x)

    def compute_class(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))

    def update_centroid(self, x):
        self.centroids = np.array([np.mean(x[self.classes == k], axis=0)  for k in range(self.K)])


