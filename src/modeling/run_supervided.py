import logging
import pandas as pd

from src.modeling.supervised import SupervisedModels
from src.modeling.stacking import StackingEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/subject_level_clustered.csv"

logger.info("Loading clustered dataset...")
df = pd.read_csv(DATA_PATH)

models = SupervisedModels(df)

dt = models.decision_tree()
rf = models.random_forest()
svm = models.svm()


stacking = StackingEnsemble(
    models.X_train,
    models.X_test,
    models.y_train,
    models.y_test
)

stacking.train()
