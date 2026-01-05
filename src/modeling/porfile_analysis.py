import pandas as pd
import logging

logger = logging.getLogger(__name__)

def analyze_clusters(df):
    logger.info("Analyzing attentional profiles...")

    summary = (
        df
        .groupby("cluster")
        .mean(numeric_only=True)
    )

    logger.info("Cluster summary:")
    logger.info(f"\n{summary}")

    return summary
