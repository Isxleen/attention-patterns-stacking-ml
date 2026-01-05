import pandas as pd
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class TrialCleaner:
    """
    Limpieza general a nivel de ensayo:
    - Eliminación de bloques de práctica
    - Eliminación de valores perdidos
    - Filtrado de latencias no plausibles

    Generic trial-level cleaning:
    - Removal of practice blocks
    - Missing value exlusion
    - Filtering of implausive latencies
    """

    def __init__(self, df):
        self.df = df.copy()

    def remove_practice_trials(self):
        if "blockcode" in self.df.columns:
            before = self.df.shape[0]
            self.df = self.df[self.df["blockcode"] != "practice"]
            after = self.df.shape[0]

            logger.info(
                f"Removed practice trials: {before - after} "
                f"(remaining: {after})"
            )
        else:
            logger.info("No blockcode column found; skipping practice removal")

        return self

    def remove_missing(self):
        critical = ["subjectid", "correct"]
        self.df = self.df.dropna(subset=critical)
        return self

    def filter_latency(self, min_rt=200, max_rt=2000):
        if "latency" in self.df.columns:
            self.df = self.df[
                (self.df["latency"] >= min_rt) &
                (self.df["latency"] <= max_rt)
            ]

        if "values.RT" in self.df.columns:
            self.df = self.df[
                (self.df["values.RT"] >= min_rt) &
                (self.df["values.RT"] <= max_rt)
            ]

        return self

    def get_clean_data(self):
        return self.df


class TaskSpecificCleaner:

    def __init__(self, df):
        self.df = df.copy()

    # -------------------------
    # FLANKER RELEVANT COLS
    # -------------------------
    def clean_flanker(self):
        cols = [
            "subjectid",
            "blocknum",
            "trialnum",
            "values.congruence",
            "response",
            "correct",
            "latency",
            "task"
        ]

        flanker = self.df[self.df["task"] == "flanker"]
        flanker = flanker[cols]

        return flanker

    # -------------------------
    # STROOP RELEVANT COLS
    # -------------------------
    def clean_stroop(self):
        cols = [
            "subjectid",
            "blocknum",
            "trialnum",
            "values.congruency",
            "stimulusitem1",
            "response",
            "correct",
            "latency",
            "task"
        ]

        stroop = self.df[self.df["task"] == "stroop"]
        stroop = stroop[cols]

        return stroop

    # -------------------------
    # SART RELEVANT COLS
    # -------------------------
    def clean_sart(self):
        cols = [
            "subjectid",
            "blocknum",
            "trialnum",
            "values.trialtype",
            "values.digit",
            "response",
            "correct",
            "values.RT",
            "values.responsetype",
            "values.latencytype",
            "task"
        ]

        sart = self.df[self.df["task"] == "sart"]
        sart = sart[cols]

        return sart


class SubjectLevelAggregator:
    """
    Construcción del dataset final:
    UNA fila por sujeto con métricas por tarea

    Final dataset construction:
    One row per subject with task-specific metrics
    """

    def __init__(self, flanker_df, stroop_df, sart_df):
        self.flanker = flanker_df
        self.stroop = stroop_df
        self.sart = sart_df

        logger.info(f"Flanker trials after cleaning: {self.flanker.shape}")
        logger.info(f"Stroop trials after cleaning: {self.stroop.shape}")
        logger.info(f"SART trials after cleaning: {self.sart.shape}")

    # -------------------------
    # FLANKER
    # -------------------------
    def aggregate_flanker(self):
        df = (
            self.flanker
            .groupby("subjectid")
            .agg(
                flanker_rt=("latency", "mean"),
                flanker_accuracy=("correct", "mean"),
                flanker_error_rate=("correct", lambda x: 1 - x.mean())
            )
            .reset_index()
        )

        logger.info(f"Flanker subjects aggregated: {df.shape[0]}")
        return df

    # -------------------------
    # STROOP
    # -------------------------
    def aggregate_stroop(self):
        df = (
            self.stroop
            .groupby("subjectid")
            .agg(
                stroop_rt=("latency", "mean"),
                stroop_accuracy=("correct", "mean"),
                stroop_error_rate=("correct", lambda x: 1 - x.mean())
            )
            .reset_index()
        )

        logger.info(f"Stroop subjects aggregated: {df.shape[0]}")
        return df

    # -------------------------
    # SART
    # -------------------------
    def aggregate_sart(self):
        df = (
            self.sart
            .groupby("subjectid")
            .agg(
                sart_rt=("values.RT", "mean"),
                sart_accuracy=("correct", "mean"),
                sart_error_rate=("correct", lambda x: 1 - x.mean())
            )
            .reset_index()
        )

        logger.info(f"SART subjects aggregated: {df.shape[0]}")
        return df

    # -------------------------
    # MERGE FINAL
    # -------------------------
    def build_final_dataset(self):
        flanker = self.aggregate_flanker()
        stroop = self.aggregate_stroop()
        sart = self.aggregate_sart()

        final_df = (
            flanker
            .merge(stroop, on="subjectid", how="inner")
            .merge(sart, on="subjectid", how="inner")
        )

        logger.info(f"FINAL dataset shape: {final_df.shape}")
        logger.info(f"FINAL columns: {list(final_df.columns)}")

        missing = final_df.isna().sum()
        if missing.any():
            logger.warning("Missing values detected in final dataset:")
            logger.warning(missing[missing > 0])
        else:
            logger.info("No missing values in final dataset")

        logger.info("Final dataset preview:")
        logger.info(f"\n{final_df.head()}")

        return final_df
