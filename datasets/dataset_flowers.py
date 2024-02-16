import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np 
import random
from datasets.prompt_dataset import PromptDataset

class Flowers(PromptDataset):
	def __init__(self, root_dir, split, resize=224):
		PromptDataset.__init__(self, root_dir, resize, split)
		self.prompt_subject = "flower breed"
		self.dataset_name = "flowers"

		self.load_data() 
		self.filter_names_to_gpt4_results()
		self.set_prompts()

	def load_data(self): 
		flowers_folders = os.listdir(os.path.join(self.root_dir, self.split))
		self.filenames = []
		self.labels = []

		# load cat_to_name.json
		import json
		with open(f'{self.root_dir}/cat_to_name.json', 'r') as f:
			cat_to_name = json.load(f)

		for flower_folder in flowers_folders:
			samples = [f"{self.root_dir}/{self.split}/{flower_folder}/{filename}" for filename in os.listdir(f"{self.root_dir}/{self.split}/{flower_folder}")]
			self.filenames.extend(samples)
			self.labels.extend([cat_to_name[flower_folder]] * len(samples))

		self.all_labels = sorted(list(set(self.labels)))
		self.all_labels = [x.lower() for x in self.all_labels]


