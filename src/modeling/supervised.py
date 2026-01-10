import logging

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

    def __init__(self, df, target="cluster", test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target = target
        self.random_state = random_state

        self.X = self.df.drop(columns=["subjectid", target])
        self.y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state
        )

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
        logger.info(f"Results for {name}")
        logger.info("\n" + classification_report(self.y_test, y_pred, zero_division=0))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(self.y_test, y_pred)))

    def decision_tree(self):
        logger.info("Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=self.random_state)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(preds, "Decision Tree")
        return self._metrics(self.y_test, preds)

    def random_forest(self):
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(preds, "Random Forest")
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
