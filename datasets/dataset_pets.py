import random
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import numpy as np 
import random
from datasets.prompt_dataset import PromptDataset

class Pets(PromptDataset):
	def __init__(self, root_dir, split, resize=224):
		PromptDataset.__init__(self, root_dir, resize, split)
		self.prompt_subject = "pet breed"
		self.dataset_name = "pets"

		self.load_labels() 
		self.filter_names_to_gpt4_results()
		self.set_prompts()

	def load_labels(self): 
		split_file = open(f"{self.root_dir}/annotations/{self.split}.txt").readlines()
		split_file_breed = [line.split(" ")[0] for line in split_file]

		pets = ["cat", "dog"]
		split_file_pet = [int(line.split(" ")[2]) for line in split_file]
		self.labels_pets = [pets[x - 1] for x in split_file_pet]

		self.filenames = [f"{self.root_dir}/images/{filename}.jpg" for filename in split_file_breed]
		self.labels = [" ".join(filename.split(".")[0].split("_")[:-1]).lower() for filename in split_file_breed]
		self.all_labels = sorted(list(set(self.labels)))
		self.all_labels = [x.lower() for x in self.all_labels]

