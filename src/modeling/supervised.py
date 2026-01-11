import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logger = logging.getLogger(__name__)


class SupervisedModels:
    """
    Modelado supervisado usando pseudo-etiquetas (clusters).
    """

    def __init__(self, df, target="cluster", test_size=0.2, random_state=42, results_dir="results"):
        self.df = df.copy()
        self.target = target
        self.random_state = random_state
        self.results_dir = results_dir  # ✅ ESTA LÍNEA FALTABA

        self.X = self.df.drop(columns=["subjectid", target])
        self.y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state
        )

        os.makedirs(self.results_dir, exist_ok=True)

    def get_splits(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def _report(self, y_pred, name):
        # nombre limpio para archivos
        safe_name = name.replace(" ", "")

        report = classification_report(self.y_test, y_pred, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)

        logger.info(f"Results for {name}")
        logger.info("\n" + report)
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(cm))

        # --- Guardar report txt ---
        report_path = os.path.join(self.results_dir, f"classification_report_{safe_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # --- Guardar CM csv ---
        cm_csv_path = os.path.join(self.results_dir, f"confusion_matrix_{safe_name}.csv")
        pd.DataFrame(cm).to_csv(cm_csv_path, index=False)

        # --- Guardar CM png ---
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        cm_png_path = os.path.join(self.results_dir, f"confusion_matrix_{safe_name}.png")
        plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    
        

    def decision_tree(self):
        logger.info("Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=self.random_state)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(preds, "DecisionTree")
        return self._metrics(self.y_test, preds)

    def random_forest(self):
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(preds, "RandomForest")
        return self._metrics(self.y_test, preds)

    def svm(self):
        logger.info("Training SVM (scaled)...")
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=self.random_state))
        ])
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(preds, "SVM")
        return self._metrics(self.y_test, preds)

    def run_all_models(self):
        logger.info("Running all supervised models...")
        return {
            "DecisionTree": self.decision_tree(),
            "RandomForest": self.random_forest(),
            "SVM": self.svm()
        }
