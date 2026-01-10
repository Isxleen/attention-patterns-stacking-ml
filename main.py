import logging
import os
import pandas as pd

from src.load_data import DataIngestor
from src.preprocessing import (
    TaskSpecificCleaner,
    TrialCleaner,
    SubjectLevelAggregator
)
from src.feature_engineering import PCAFeatureExtractor

from src.modeling.unsupervised import UnsupervisedModeling
from src.modeling.supervised import SupervisedModels
from src.modeling.stacking import StackingEnsemble


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

SUBJECT_DATASET_FILE = "subject_level_dataset.csv"
PCA_DATASET_FILE = "subject_level_pca.csv"
CLUSTERED_DATASET_FILE = "subject_level_clustered.csv"

# parámetros simples (como lo haces tú)
USE_PCA_FOR_CLUSTERING = False   # si quieres clusterizar con PCA
N_CLUSTERS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # -----------------------------------------------------
    # 1. Carga de datos
    # -----------------------------------------------------
    logger.info("Loading raw datasets...")
    ingestor = DataIngestor(RAW_DATA_PATH)
    raw_df = ingestor.load_and_merge()
    logger.info(f"Raw dataset shape: {raw_df.shape}")

    # -----------------------------------------------------
    # 2. Eliminación de ensayos de práctica
    # -----------------------------------------------------
    logger.info("Removing practice trials from raw data...")
    raw_df = (
        TrialCleaner(raw_df)
        .remove_practice_trials()
        .get_clean_data()
    )

    # -----------------------------------------------------
    # 3. Selección de variables por tarea
    # -----------------------------------------------------
    logger.info("Selecting task-specific variables...")
    selector = TaskSpecificCleaner(raw_df)

    flanker_df = selector.clean_flanker()
    stroop_df = selector.clean_stroop()
    sart_df = selector.clean_sart()

    # -----------------------------------------------------
    # 4. Limpieza y depuración de ensayos
    # -----------------------------------------------------
    logger.info("Cleaning Flanker trials...")
    flanker_df = (
        TrialCleaner(flanker_df)
        .remove_missing()
        .filter_latency()
        .get_clean_data()
    )

    logger.info("Cleaning Stroop trials...")
    stroop_df = (
        TrialCleaner(stroop_df)
        .remove_missing()
        .filter_latency()
        .get_clean_data()
    )

    logger.info("Cleaning SART trials...")
    sart_df = (
        TrialCleaner(sart_df)
        .remove_missing()
        .filter_latency()
        .get_clean_data()
    )

    logger.info(f"Flanker cleaned shape: {flanker_df.shape}")
    logger.info(f"Stroop cleaned shape: {stroop_df.shape}")
    logger.info(f"SART cleaned shape: {sart_df.shape}")

    # -----------------------------------------------------
    # 5. Agregación a nivel de participante
    # -----------------------------------------------------
    logger.info("Aggregating subject-level dataset...")
    aggregator = SubjectLevelAggregator(
        flanker_df=flanker_df,
        stroop_df=stroop_df,
        sart_df=sart_df
    )

    final_df = aggregator.build_final_dataset()

    subject_output_path = os.path.join(PROCESSED_DATA_PATH, SUBJECT_DATASET_FILE)
    final_df.to_csv(subject_output_path, index=False)
    logger.info(f"Subject-level dataset saved to: {subject_output_path}")

    # -----------------------------------------------------
    # 6. PCA – Feature Matrix final
    # -----------------------------------------------------
    logger.info("Applying PCA to subject-level dataset...")
    pca_extractor = PCAFeatureExtractor(final_df, variance_threshold=0.95)
    pca_df = pca_extractor.get_pca_dataframe()

    pca_output_path = os.path.join(PROCESSED_DATA_PATH, PCA_DATASET_FILE)
    pca_df.to_csv(pca_output_path, index=False)

    logger.info(f"PCA dataset shape: {pca_df.shape}")
    logger.info(f"PCA dataset saved to: {pca_output_path}")


    # -----------------------------------------------------
    # 7. UNSUPERVISED (clustering -> pseudo-etiquetas)
    # -----------------------------------------------------
    logger.info("Running unsupervised clustering...")

    # IMPORTANTE:
    # UnsupervisedModeling normalmente espera el df "original" y dentro decide si usa PCA.
    # Si tu clase ya hace PCA internamente con `use_pca=True`, pásale final_df.
    # Si tu clase espera que le pases ya el df con PCA, pásale pca_df.
    df_for_clustering = final_df  # por defecto (más seguro si tu clase hace PCA dentro)

    unsup = UnsupervisedModeling(df_for_clustering, use_pca=USE_PCA_FOR_CLUSTERING)
    labels, sil_score = unsup.kmeans_clustering(n_clusters=N_CLUSTERS)
    clustered_df = unsup.build_clustered_dataset(labels)

    clustered_output_path = os.path.join(PROCESSED_DATA_PATH, CLUSTERED_DATASET_FILE)
    clustered_df.to_csv(clustered_output_path, index=False)

    logger.info(f"Clustered dataset saved to: {clustered_output_path}")
    logger.info(f"Silhouette score: {sil_score}")

    # -----------------------------------------------------
    # 8. SUPERVISED (entrenamiento con pseudo-etiquetas)
    # -----------------------------------------------------
    logger.info("Training supervised models...")
    sup = SupervisedModels(
        clustered_df,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE
    )

    results = sup.run_all_models()
    logger.info(f"Supervised results: {results}")

    # -----------------------------------------------------
    # 9. STACKING (opcional)
    # -----------------------------------------------------
    logger.info("Training stacking ensemble...")
    X_train, X_test, y_train, y_test = sup.get_splits()
    stack = StackingEnsemble(X_train, X_test, y_train, y_test)
    _ = stack.train()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
