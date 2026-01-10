import logging
import numpy as np

from collections import Counter
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


logger = logging.getLogger(__name__)


class StackingEnsemble:
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state

    def train(self):
        logger.info("Training Stacking Ensemble...")

        min_class = min(Counter(self.y_train).values())
        cv = min(5, min_class)  # no puede ser mayor que la clase minoritaria
        cv = max(2, cv)         # mínimo 2 para que tenga sentido

        estimators = [
            ("dt", DecisionTreeClassifier(random_state=self.random_state)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)),
            ("svm", Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", probability=True, random_state=42)) ])
            )
        ]

        meta_model = LogisticRegression(max_iter=2000)

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=-1
        )

        stacking.fit(self.X_train, self.y_train)
        preds = stacking.predict(self.X_test)

        logger.info("Stacking results:")
        logger.info("\n" + classification_report(self.y_test, preds, zero_division=0))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(self.y_test, preds)))

        return stacking
