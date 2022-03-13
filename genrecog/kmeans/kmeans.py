from sklearn.cluster import KMeans
import numpy as np


class Kmeans:
    def __init__(
            self,
            clusters=10,
            verbose=0

    ):
        self.X = []
        self.model = KMeans(
            n_clusters=clusters,
            verbose=verbose
        )



