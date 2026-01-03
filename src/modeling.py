import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class UnsupervisedModeling:
    """
    Pipeline de modelado no supervisado para la identificación
    de perfiles atencionales latentes

    Unsupervised modeling pipeline for the identification
    of lattent attentional profiles
    """

    def __init__(self, df, use_pca=False, n_components=0.95):
        self.df = df.copy()
        self.use_pca = use_pca
        self.n_components = n_components

        self.subject_ids = self.df["subjectid"]
        self.X = self.df.drop(columns=["subjectid"])

        self.scaler = StandardScaler()
        self.pca = None
    
    def scale_features(self):
        logger.info("Normalizing features (StandardScaler)...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled
    
    def apply_pca(self):
        logger.info("Appling PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        explained = np.sum(self.pca.explained_variance_ratio_) * 100
        logger.info(
            f"PCA retained {self.X_pca.shape[1]} components "
            f"explaining {explained:.2f}% variance"
        )

        return self.X_pca
    
    def get_feature_matrix(self):
        self.scale_features()
        if self.use_pca:
            return self.apply_pca()
        return self.X_scaled
    
    def kmeans_clustering(self, n_clusters):
        logger.info(f"Running K-Means(k={n_clusters})...")
        X = self.get_feature_matrix()

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        logger.info(f"K-Means silhouette score: {score:.3f}")

        return labels, score
    
    def hierarchical_clustering(self, n_clusters):
        logger.info(f"Running Hiperarchical Clustering (k={n_clusters})...")
        X = self.get_feature_matrix()

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        logger.info(f"Hierarchical silhouette score: {score:.3f}")

        return labels, score

    def build_clustered_dataset(self, labels):
        clustered_df = self.df.copy()
        clustered_df["cluster"] = labels
        return clustered_df


