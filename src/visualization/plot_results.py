import pandas as pd
from src.modeling.unsupervised import UnsupervisedModeling
from src.visualization.clustering_plots import plot_silhouette_curve
from src.visualization.profile_plots import plot_cluster_profiles

df = pd.read_csv("data/processed/subject_level_clustered.csv")

unsup = UnsupervisedModeling(df, use_pca=False)
X = unsup.scale_features()
X_pca = unsup.apply_pca()

# # PCA
# plot_scree(unsup.pca, "results/scree_plot.png")
# plot_pca_scatter(X_pca, df["cluster"], "results/pca_scatter.png")

# Clustering quality
plot_silhouette_curve(X_pca, save_path="results/silhouette_curve.png")

# Cluster profiles
features = df.drop(columns=["subjectid", "cluster"]).columns
plot_cluster_profiles(df, features, "results/cluster_profiles.png")
