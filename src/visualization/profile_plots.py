import matplotlib.pyplot as plt
import pandas as pd

def plot_cluster_profiles(df, feature_cols, save_path=None):
    grouped = df.groupby("cluster")[feature_cols].mean()

    grouped.T.plot(kind="bar")
    plt.ylabel("Valor medio")
    plt.title("Perfil cognitivo medio por cluster")
    plt.legend(title="Cluster")

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
