import matplotlib.pyplot as plt

def plot_cluster_profiles(df, feature_cols, save_path=None, show=False):
    grouped = df.groupby("cluster")[feature_cols].mean()

    ax = grouped.T.plot(kind="bar")
    ax.set_ylabel("Valor medio")
    ax.set_title("Perfil cognitivo medio por cluster")
    ax.legend(title="Cluster")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
