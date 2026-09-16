from router.config import load_config
from training.data.facade import load_training_data


d = load_training_data(load_config())
d.correctness_matrix(split="train").shape