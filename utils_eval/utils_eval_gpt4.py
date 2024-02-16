import random
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import requests
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import seaborn as sns

import cv2
import numpy as np
import os
from tqdm import tqdm
import string

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import textwrap
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from utils import *

def run_exp_base(dataset, dataset_name = "cars", image_int = 0): 
	from utils_models.utils_gpt4 import get_gpt4_pred

	for data in tqdm(dataset): 

		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)
		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue

		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_base.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text(dataset, opt, dataset_name = "cars", text_position="top", image_int = 0): 
	from utils_models.utils_gpt4 import get_gpt4_pred
 	
	random_data = get_random_data(dataset_name)

	for data in tqdm(dataset): 

		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		wrong_label = random_data["wrong_labels_new_all"][random_data["image_names_all"].index(image_name)]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		image = insert_text(image, wrong_label, image_height, color="black", font_size = 40, position=text_position) 
		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue

		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_gpt4_attack(dataset, opt, dataset_name = "cars", text_position="top", describe_position="bottom", image_int = 0, insert_text_check=True, insert_text_desc_check=True, random_desc=False): 
	from utils_models.utils_gpt4 import get_gpt4_pred
	
	gpt4_data = get_gpt4_data(dataset_name)

	for data in tqdm(dataset): 

		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		wrong_label_new = gpt4_data["wrong_labels_new_all"][gpt4_data["image_names_all"].index(image_name)]

		if random_desc:
			description = random.choice(gpt4_data["descriptions_all"])
		else: 
			description = gpt4_data["descriptions_all"][gpt4_data["image_names_all"].index(image_name)]

		if insert_text_check:
			image = insert_text(image, wrong_label_new, image_height, color="black", font_size = 40, position=text_position) 

		if insert_text_desc_check:
			image = insert_text(image, description, image_height, color="black", font_size = 25, position=describe_position) 

		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue
			
		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")		

def run_exp_base_text_gpt4_attack_llm(dataset, opt, dataset_name = "cars", text_position="top", image_int = 0, insert_text_check=True): 
	from utils_models.utils_gpt4 import get_gpt4_pred
	
	gpt4_data = get_gpt4_data_llm(dataset_name)

	for data in tqdm(dataset): 


		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]


		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		wrong_label_new = gpt4_data["wrong_labels_new_all"][gpt4_data["gt_labels"].index(label)]
		description = gpt4_data["descriptions_all"][gpt4_data["gt_labels"].index(label)]

		if insert_text_check:
			image = insert_text(image, wrong_label_new, image_height, color="black", font_size = 40, position=text_position) 
		
		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue

		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_clip_attack(dataset, opt, dataset_name = "cars", text_position="top", describe_position="bottom", image_int = 0, insert_text_check=True, insert_text_desc_check=True): 
	from utils_models.utils_gpt4 import get_gpt4_pred
	
	clip_data = get_clip_data(dataset_name)

	for data in tqdm(dataset): 

		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		wrong_label_new = clip_data["wrong_labels_new_all"][clip_data["image_names"].index(image_name)]

		if insert_text_check:
			image = insert_text(image, wrong_label_new, image_height, color="black", font_size = 40, position=text_position) 

		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue
		
		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")		


		
		
def run_exp_base_text_llama_attack_llm(dataset, opt, dataset_name = "cars", text_position="top", image_int = 0, insert_text_check=True): 
	from utils_models.utils_gpt4 import get_gpt4_pred
	
	llama_data = get_llama_data_llm(dataset_name)

	for data in tqdm(dataset): 


		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		wrong_label_new = llama_data["wrong_labels_new_all"][llama_data["gt_labels"].index(label)]

		if insert_text_check:
			image = insert_text(image, wrong_label_new, image_height, color="black", font_size = 40, position=text_position) 

		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue

		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_llava_attack(dataset, opt, dataset_name = "cars", text_position="top", image_int = 0, insert_text_check=True): 
	from utils_models.utils_gpt4 import get_gpt4_pred
	
	llava_data = get_llava_data(dataset_name)

	for data in tqdm(dataset): 


		image_name = data["image_name"]
		image = data["image"]
		prompt = data["prompt"]
		answer = data["answer"]
		label = data["label"]

		image = Image.fromarray(image).convert("RGB")
		image, image_height = add_white_background(image)

		wrong_label_new = llava_data["wrong_labels_new_all"][llava_data["image_names"].index(image_name)]

		if insert_text_check:
			image = insert_text(image, wrong_label_new, image_height, color="black", font_size = 40, position=text_position) 

		image = image.resize((224, 224))

		image.save(f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg")
		image_path = f"outputs/temp/image_{dataset_name}_{text_position}_{opt.method}.jpg"

		try: 
			out = get_gpt4_pred(image_path, prompt)
			out = out.replace(",", " ").replace("\n", " ")
		except: 
			continue

		data_to_write = [image_name, out, answer]
		with open(f"outputs/output_gpt4/{dataset_name}_{opt.method}.txt", "a") as f: 
			f.write(",".join([str(x) for x in data_to_write]) + "\n")
