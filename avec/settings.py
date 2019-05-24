from os.path import dirname, join

ROOT = join(dirname(dirname(__file__)))

PATH_TO_DEVELOPMENT_DATA = join(ROOT, "data", "validation")
PATH_TO_TRAINING_DATA = join(ROOT, "data", "train")
PATH_TO_TEST_DATA = join(ROOT, "data", "test")