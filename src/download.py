from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ds = load_dataset("lmarena-ai/arena-human-preference-55k")
ds.save_to_disk("data/raw/arena-human-preference-55k")
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("routellm/gpt4_judge_battles")
ds.save_to_disk("data/raw/gpt4_judge_battles")