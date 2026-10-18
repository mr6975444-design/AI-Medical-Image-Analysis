from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "chest_xray"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "medical_image_classifier_class_weighted.keras"

OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"


# ============================================================
# IMAGE / TRAINING SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 5

SEED = 42


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)
