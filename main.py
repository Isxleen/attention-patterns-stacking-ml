from src.load_data import DataIngestor
from src.preprocessing import (
    TaskSpecificCleaner,
    TrialCleaner,
    SubjectLevelAggregator
)

import logging
import os

# -----------------------------------------------------
# Configuración básica de logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Rutas
# -----------------------------------------------------
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
OUTPUT_FILE = "subject_level_dataset.csv"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# -----------------------------------------------------
# 1. Carga de datos 
# -----------------------------------------------------
logger.info("Loading raw datasets...")
ingestor = DataIngestor(RAW_DATA_PATH)
raw_df = ingestor.load_and_merge()
logger.info(f"Raw dataset shape: {raw_df.shape}")

# -----------------------------------------------------
# 2. Selección de variables por tarea
# -----------------------------------------------------
raw_df = (
    TrialCleaner(raw_df)
    .remove_practice_trials()
    .get_clean_data()
)
logger.info("Selecting task-specific variables...")
selector = TaskSpecificCleaner(raw_df)

flanker_df = selector.clean_flanker()
stroop_df = selector.clean_stroop()
sart_df = selector.clean_sart()

# -----------------------------------------------------
# 3. Limpieza y depuración de ensayos (3.4.1)
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
    .remove_practice_trials()
    .remove_missing()
    .filter_latency()
    .get_clean_data()
)

logger.info(f"Flanker cleaned shape: {flanker_df.shape}")
logger.info(f"Stroop cleaned shape: {stroop_df.shape}")
logger.info(f"SART cleaned shape: {sart_df.shape}")

# -----------------------------------------------------
# 4. Agregación por participante 
# -----------------------------------------------------
logger.info("Aggregating subject-level dataset...")
aggregator = SubjectLevelAggregator(
    flanker_df=flanker_df,
    stroop_df=stroop_df,
    sart_df=sart_df
)

final_df = aggregator.build_final_dataset()

# -----------------------------------------------------
# 5. Guardado del dataset final
# -----------------------------------------------------
output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)
final_df.to_csv(output_path, index=False)

logger.info(f"Final dataset saved to: {output_path}")
logger.info("Preprocessing pipeline completed successfully.")
