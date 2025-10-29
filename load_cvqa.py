from datasets import load_dataset

dataset = load_dataset("afaji/cvqa", split="test")
argentina = dataset.filter(lambda x: x["Subset"] == "('Portuguese', 'Brazil')", num_proc=4)
argentina.save_to_disk("cvqa_brazil")