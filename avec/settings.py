from os.path import dirname, join

ROOT = dirname(dirname(__file__))

PATH_TO_VALIDATION_DATA = join(ROOT, "data", "validation")
PATH_TO_TRAINING_DATA = join(ROOT, "data", "train")
PATH_TO_TEST_DATA = join(ROOT, "data", "test")

PATH_TO_MIN_MAX_SCALER = join(ROOT, "models", "scikit",
        "min_max_scaler_training.pickle")

OPEN_FACE_STATE_DICT = join(ROOT, "models", "lstm", "open_face_model.pt")
RESNET_STATE_DICT = join(ROOT, "models", "lstm", "resnet_model.pt")

# PATH_TO_LABELS = join(ROOT, "data", "labels.csv")