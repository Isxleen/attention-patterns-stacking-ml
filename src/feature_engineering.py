import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PCAFeatureExtractor:
    """
    Aplicación de PCA sobre el dataset final a nivel de sujeto.
    """

    def __init__(self, df, variance_threshold=0.95):
        self.df = df.copy()
        self.variance_threshold = variance_threshold

        self.subject_ids = self.df["subjectid"]
        self.features = self.df.drop(columns=["subjectid"])

        self.scaler = StandardScaler()
        self.pca = None

    def scale_features(self):
        """
        Estandariza las variables numéricas
        """
        logger.info("Standardizing features...")
        scaled = self.scaler.fit_transform(self.features)
        return scaled

    def fit_pca(self):
        """
        Ajusta PCA reteniendo la varianza deseada
        """
        scaled_data = self.scale_features()

        self.pca = PCA(n_components=self.variance_threshold)
        components = self.pca.fit_transform(scaled_data)

        logger.info(
            f"PCA fitted: {self.pca.n_components_} components "
            f"explain {np.sum(self.pca.explained_variance_ratio_):.2%} variance"
        )

        return components

    def get_pca_dataframe(self):
        """
        Devuelve DataFrame con componentes principales
        """
        components = self.fit_pca()

        columns = [
            f"PC{i+1}" for i in range(components.shape[1])
        ]

        pca_df = pd.DataFrame(
            components,
            columns=columns
        )

        pca_df.insert(0, "subjectid", self.subject_ids.values)

        logger.info(f"PCA dataset shape: {pca_df.shape}")
        logger.info("PCA dataset preview:")
        logger.info(f"\n{pca_df.head()}")

        return pca_df

    def get_loadings(self):
        """
        Devuelve las cargas de cada variable en los componentes
        """
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(self.pca.n_components_)],
            index=self.features.columns
        )

        logger.info("PCA loadings:")
        logger.info(f"\n{loadings}")

        return loadings

    def get_explained_variance(self):
        """
        Devuelve la varianza explicada por componente
        """
        variance = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f"PC{i+1}" for i in range(self.pca.n_components_)]
        )

        logger.info("Explained variance by component:")
        logger.info(f"\n{variance}")

        return variance
    
    def plot_scree_plot(self):
        """
        Genera y muestra el Scree Plot (Gráfico de sedimentación)
        """
        if self.pca is None:
            logger.error("PCA has not been fitted yet. Call fit_pca() or get_pca_dataframe() first.")
            return

        explained_variance = self.pca.explained_variance_ratio_
        
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(explained_variance) + 1), 
            explained_variance, 
            marker='o', 
            linestyle='--', 
            color='b'
        )
        
        # Opcional: Añadir la varianza acumulada para ver el progreso hacia el 95%
        plt.step(
            range(1, len(explained_variance) + 1), 
            np.cumsum(explained_variance), 
            where='mid', 
            label='Cumulative Explained Variance',
            alpha=0.5
        )

        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot - Varianza Explicada por Componente")
        plt.xticks(range(1, len(explained_variance) + 1)) # Asegura que salgan números enteros en el eje X
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.show()
