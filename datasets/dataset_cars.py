import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np 
import random
from datasets.prompt_dataset import PromptDataset

class Cars(PromptDataset):
	def __init__(self, root_dir, split, resize=224):
		PromptDataset.__init__(self, root_dir, resize, split)
		self.prompt_subject = "car model"
		self.dataset_name = "cars"

		if self.split == "train":
			self.anno_path = os.path.join(self.root_dir, "cars_train_annos.mat")
			self.img_path = os.path.join(self._base_folder, "cars_train", "cars_train")
		else:
			self.anno_path = os.path.join(self.root_dir, "cars_test_annos_withlabels.mat")
			self.img_path = os.path.join(self.root_dir, "cars_test", "cars_test")

		self.load_data() 
		self.filter_names_to_gpt4_results()
		self.set_prompts()

	def load_data(self): 
		self.anno_path = os.path.join(self.anno_path)
		annos_data = sio.loadmat(self.anno_path, squeeze_me=True)
		annos_classes = sio.loadmat(os.path.join(self.root_dir, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()

		annos = annos_data["annotations"]
		class_names = [x.split(" ")[0].lower().replace("-", "") for x in annos_classes]

		self.filenames = [os.path.join(self.img_path, annotation["fname"]) for annotation in annos]
		self.labels = [class_names[annotation["class"] - 1] for annotation in annos]
		self.all_labels = sorted(list(set(self.labels)))
		self.all_labels = [x.lower() for x in self.all_labels]

