import logging
import pandas as pd

logger = logging.getLogger(__name__)


class TrialCleaner:
    """
    Limpieza general a nivel de ensayo:
    - Eliminación de bloques de práctica
    - Eliminación de valores perdidos
    - Filtrado de latencias no plausibles
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_practice_trials(self):
        if "blockcode" in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df["blockcode"] != "practice"]
            after = len(self.df)
            logger.info(f"Removed practice trials: {before - after} (remaining: {after})")
        return self

    def ensure_correct_numeric(self):
        # Convierte bool -> int, "true/false" -> NaN si no se puede
        if "correct" in self.df.columns:
            self.df["correct"] = pd.to_numeric(self.df["correct"], errors="coerce")
        return self

    def remove_missing(self):
        critical = ["subjectid", "correct"]
        self.df = self.df.dropna(subset=critical)
        return self

    def filter_latency(self, min_rt=200, max_rt=2000):
        if "latency" in self.df.columns:
            self.df = self.df[(self.df["latency"] >= min_rt) & (self.df["latency"] <= max_rt)]

        if "values.RT" in self.df.columns:
            self.df = self.df[(self.df["values.RT"] >= min_rt) & (self.df["values.RT"] <= max_rt)]

        return self

    def get_clean_data(self) -> pd.DataFrame:
        return self.df


class TaskSpecificCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _select(self, task_name: str, cols: list[str]) -> pd.DataFrame:
        task_df = self.df[self.df["task"] == task_name]
        missing_cols = [c for c in cols if c not in task_df.columns]
        if missing_cols:
            raise KeyError(f"Missing columns for {task_name}: {missing_cols}")
        return task_df[cols]

    def clean_flanker(self):
        cols = ["subjectid", "blocknum", "trialnum", "values.congruence", "response", "correct", "latency", "task"]
        return self._select("flanker", cols)

    def clean_stroop(self):
        cols = ["subjectid", "blocknum", "trialnum", "values.congruency", "stimulusitem1",
                "response", "correct", "latency", "task"]
        return self._select("stroop", cols)

    def clean_sart(self):
        cols = ["subjectid", "blocknum", "trialnum", "values.trialtype", "values.digit",
                "response", "correct", "values.RT", "values.responsetype", "values.latencytype", "task"]
        return self._select("sart", cols)


class SubjectLevelAggregator:
    """
    Una fila por sujeto con métricas por tarea.
    """

    def __init__(self, flanker_df: pd.DataFrame, stroop_df: pd.DataFrame, sart_df: pd.DataFrame):
        self.flanker = flanker_df
        self.stroop = stroop_df
        self.sart = sart_df

        logger.info(f"Flanker trials after cleaning: {self.flanker.shape}")
        logger.info(f"Stroop trials after cleaning: {self.stroop.shape}")
        logger.info(f"SART trials after cleaning: {self.sart.shape}")

    def aggregate_flanker(self):
        df = self.flanker.groupby("subjectid").agg(
            flanker_rt=("latency", "mean"),
            flanker_accuracy=("correct", "mean"),
        ).reset_index()
        df["flanker_error_rate"] = 1 - df["flanker_accuracy"]
        return df

    def aggregate_stroop(self):
        df = self.stroop.groupby("subjectid").agg(
            stroop_rt=("latency", "mean"),
            stroop_accuracy=("correct", "mean"),
        ).reset_index()
        df["stroop_error_rate"] = 1 - df["stroop_accuracy"]
        return df

    def aggregate_sart(self):
        df = self.sart.groupby("subjectid").agg(
            sart_rt=("values.RT", "mean"),
            sart_accuracy=("correct", "mean"),
        ).reset_index()
        df["sart_error_rate"] = 1 - df["sart_accuracy"]
        return df

    def build_final_dataset(self):
        final_df = (
            self.aggregate_flanker()
            .merge(self.aggregate_stroop(), on="subjectid", how="inner")
            .merge(self.aggregate_sart(), on="subjectid", how="inner")
        )

        logger.info(f"FINAL dataset shape: {final_df.shape}")
        return final_df
