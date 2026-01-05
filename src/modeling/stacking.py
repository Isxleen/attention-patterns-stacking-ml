import logging

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class StackingEnsemble:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train(self):
        logger.info("Training Stacking Ensemble..")
        estimators = [
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ]

        meta_model = LogisticRegression(max_iter=1000)
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        stacking.fit(self.X_train, self.y_train)
        preds = stacking.predict(self.X_test)

        logger.info("Stacking results:")
        logger.info("\n" + classification_report(self.y_test, preds))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(self.y_test, preds)))

        return stacking