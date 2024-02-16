import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json 
import pandas as pd
from datasets.prompt_dataset import PromptDataset


class Food101(PromptDataset):
	def __init__(self, root_dir, resize=224, split="test"):
		PromptDataset.__init__(self, root_dir, resize, split)
		self.prompt_subject = "food"
		self.dataset_name = "food101"

		self.load_data() 
		self.filter_names_to_gpt4_results()
		self.set_prompts()

	def load_data(self):
		split_data = json.load(open(f"{self.root_dir}/meta/{self.split}.json"))
		classes_folders = list(split_data.keys())

		self.filenames = []
		self.labels = []

		for class_folder in classes_folders:
			samples = [f"{self.root_dir}/images/{filename}.jpg" for filename in split_data[class_folder]]

			self.filenames.extend(samples)
			self.labels.extend([class_folder.replace("_", " ")] * len(samples))

		self.all_labels = sorted(list(set(self.labels)))
		self.all_labels = [x.lower() for x in self.all_labels]
