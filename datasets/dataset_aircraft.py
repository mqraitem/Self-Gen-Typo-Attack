import os
import json
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from datasets.prompt_dataset import PromptDataset

class Aircraft(PromptDataset) :
	def __init__(self, root_dir, resize=224, split="test") :
		PromptDataset.__init__(self, root_dir, resize, split)
		self.prompt_subject = "aircraft model"
		self.dataset_name = "aircraft"

		self.load_data()
		self.filter_names_to_gpt4_results()
		self.set_prompts()

	def load_data(self) :
		aircraft_folder = os.path.join(self.root_dir, "data", "images")
		with open(f'{self.root_dir}/aircraft_labels.json', 'r') as f:
			aircraft_to_family = json.load(f)
		aircraft_keys = list(aircraft_to_family.keys())

		split_file = os.path.join(self.root_dir, "data",f"images_{self.split}.txt")
		samples = open(split_file).read().splitlines()
		self.filenames = [os.path.join(aircraft_folder, f"{filename}.jpg") for filename in samples]        
		self.labels = [aircraft_to_family[filename] for filename in samples]
		self.labels = [x.lower() for x in self.labels]

		self.all_labels = sorted(list(set(self.labels)))
		self.all_labels = [x.lower() for x in self.all_labels]



