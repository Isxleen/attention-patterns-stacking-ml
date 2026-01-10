import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def plot_silhouette_curve(X, max_k=8, save_path=None):
    scores = []

    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    plt.figure()
    plt.plot(range(2, max_k + 1), scores, marker="o")
    plt.xlabel("Número de klusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Evaluación del número de clusters")

    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()