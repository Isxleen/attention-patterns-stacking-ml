# src/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Paths
    RAW_DIR: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    RUNS_DIR: str = "results/runs"

    # Raw filenames (los tuyos)
    FLANKER_FILE: str = "Raw_Flanker.xlsx"
    STROOP_FILE: str = "Raw_Stroop.xlsx"
    SART_FILE: str = "Raw_Sart.xlsx"

    # Preprocesado
    MIN_RT: int = 200
    MAX_RT: int = 2000

    # Unsupervised
    USE_PCA: bool = True
    DEFAULT_K: int = 3

    # Supervised
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
