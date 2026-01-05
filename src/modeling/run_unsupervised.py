import pandas as pd
import logging
import os

from src.modeling.unsupervised import UnsupervisedModeling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed"
INPUT_FILE = "subject_level_dataset.csv"
OUTPUT_FILE = "subject_level_with_clusters.csv"

logger.info("Loading subject-level dataset...")
df = pd.read_csv(os.path.join(DATA_PATH, INPUT_FILE))

model = UnsupervisedModeling(
    df,
    use_pca=False,      
    n_components=0.95
)

# -----------------------------------------------------
# Search best k using K-Means
# -----------------------------------------------------
results = {}

for k in range(2, 7):
    labels, score = model.kmeans_clustering(n_clusters=k)
    results[k] = score

best_k = max(results, key=results.get)
logger.info(f"Best k according to silhouette: {best_k}")

final_labels, _ = model.kmeans_clustering(n_clusters=best_k)
clustered_df = model.build_clustered_dataset(final_labels)

# -----------------------------------------------------
# Save clustered dataset
# -----------------------------------------------------
output_path = os.path.join(DATA_PATH, OUTPUT_FILE)
clustered_df.to_csv(output_path, index=False)

logger.info(f"Clustered dataset saved to: {output_path}")
