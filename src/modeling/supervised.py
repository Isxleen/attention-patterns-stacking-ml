import logging
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class SupervisedModels:

    def __init__(self, df, target="cluster", test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target = target

        self.X = self.df.drop(columns=["subjectid", target])
        self.y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state
        )
    
    # -------------------------
    # DECISION TREE
    # -------------------------
    def desision_tree(self):
        logger.info("Training Decision Tree...")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(model, preds, "Decision Tree")

        return model
    
    # -------------------------
    # RANDOM FOREST
    # -------------------------
    def random_forest(self):
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(model, preds, "Random Forest")

        return model
    
     # -------------------------
    # SVM
    # -------------------------
    def svm(self):
        logger.info("Training SVM...")
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        self._report(model, preds, "SVM")

        return model

    # -------------------------
    # REPORT
    # -------------------------
    def _report(self, model, preds, name):
        logger.info(f"Results for {name}")
        logger.info("\n" + classification_report(self.y_test, preds))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(self.y_test, preds)))
