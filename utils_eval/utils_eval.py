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


def run_exp_base(data_loader, dataset_name,  model_data, opt): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if model_data["name"] in ["minigpt4", "minigpt4_vicuna"]:
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		prompts = data_batch["prompt"]
		labels = data_batch["label"]
		answers = data_batch["answer"]

		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] in ["minigpt4", "minigpt4_vicuna"]:
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue


		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]
		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")

def run_exp_base_text(data_loader, dataset_name,  model_data, opt, position="top", insert_text_check=True): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	attack_data = get_random_data(dataset_name)


	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		wrong_labels = [attack_data["wrong_labels_new_all"][attack_data["image_names_all"].index(image_name)] for image_name in image_names]
		if insert_text_check:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels, images_heights)]		
					
		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_gpt4_attack(data_loader, dataset_name,  model_data, opt, position="top", position_desc="bottom", insert_text_check=True, insert_text_desc_check=True, random_desc=False,font_desc=25): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	gpt4_data = get_gpt4_data(dataset_name)

	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]
		
		wrong_labels_new = [gpt4_data["wrong_labels_new_all"][gpt4_data["image_names_all"].index(image_name)] for image_name in image_names]
		
		if random_desc:
			description_attacks = [random.choice(gpt4_data["descriptions_all"]) for image_name in image_names]
		else:
			description_attacks = [gpt4_data["descriptions_all"][gpt4_data["image_names_all"].index(image_name)] for image_name in image_names]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if insert_text_check == True:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels_new, images_heights)]

		if insert_text_desc_check == True:		
			images = [insert_text(image, description, image_height, color=opt.color, font_size = font_desc, position=position_desc) for image, description, image_height in zip(images, description_attacks, images_heights)]		


		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")

def run_exp_base_text_gpt4_attack_llm(data_loader, dataset_name,  model_data, opt, position="top", position_desc="bottom", insert_text_check=True, insert_text_desc_check=True): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	gpt4_data = get_gpt4_data_llm(dataset_name)


	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]
		
		wrong_labels_new = [gpt4_data["wrong_labels_new_all"][gpt4_data["gt_labels"].index(label)] for label in labels]
		description_attacks = [gpt4_data["descriptions_all"][gpt4_data["gt_labels"].index(label)] for label in labels]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if insert_text_check == True:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels_new, images_heights)]

		if insert_text_desc_check == True:		
			images = [insert_text(image, description, image_height, color=opt.color, font_size = 25, position=position_desc) for image, description, image_height in zip(images, description_attacks, images_heights)]		

		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_llama_attack_llm(data_loader, dataset_name,  model_data, opt, position="top", insert_text_check=True): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	llama_data = get_llama_data_llm(dataset_name)

	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]
		
		wrong_labels_new = [llama_data["wrong_labels_new_all"][llama_data["gt_labels"].index(label)] for label in labels]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if insert_text_check == True:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels_new, images_heights)]

		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")

def run_exp_base_text_llava_attack(data_loader, dataset_name,  model_data, opt, position="top", insert_text_check=True): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	llava_data = get_llava_data(dataset_name)

	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]
		
		wrong_labels_new = [llava_data["wrong_labels_new_all"][llava_data["image_names"].index(image_name)] for image_name in image_names]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if insert_text_check == True:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels_new, images_heights)]


		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")


def run_exp_base_text_clip_attack(data_loader, dataset_name,  model_data, opt, position="top", insert_text_check=True): 

	name = model_data["name"]
	if name == "blip":
		from utils_models.utils_instructblip import run_blip 

	if name == "llava":
		from utils_models.utils_llava import run_llava 

	if name == "minigpt4": 
		from utils_models.utils_minigpt4 import run_minigpt4 

	if name == "otter":
		from utils_models.utils_otter import run_otter

	clip_data = get_clip_data(dataset_name)

	for data_batch in tqdm(data_loader): 

		image_names = data_batch["image_name"]
		labels = data_batch["label"]
		prompts = data_batch["prompt"]
		answers = data_batch["answer"]
		images = [Image.fromarray(image.numpy()).convert("RGB") for image in data_batch['image']]
		
		wrong_labels_new = [clip_data["wrong_labels_new_all"][clip_data["image_names"].index(image_name)] for image_name in image_names]

		images_heights = [add_white_background(image) for image in images]
		images = [image[0] for image in images_heights]
		images_heights = [image[1] for image in images_heights]

		if insert_text_check == True:
			images = [insert_text(image, wrong_label, image_height, color=opt.color, font_size = 40, position=position) for image, wrong_label, image_height in zip(images, wrong_labels_new, images_heights)]

		if model_data["name"] == 'blip':
			text_output = run_blip(prompts, images, model_data)

		elif model_data["name"] == "llava":
			text_output = run_llava(prompts, images, model_data)

		elif model_data["name"] == "minigpt4":
			text_output = run_minigpt4(prompts, images, model_data)

		elif model_data["name"] == "otter":
			text_output = run_otter(prompts, images, model_data)

		if text_output is None:
			continue

		text_output = [x.replace(",", " ").replace("\n", " ") for x in text_output]

		with open(f"outputs/output_{model_data['name']}/{dataset_name}_{opt.method}.txt", "a") as f: 
			for image_name, text, answer in zip(image_names, text_output, answers):
				data_to_write = [image_name, text, answer]
				f.write(",".join([str(x) for x in data_to_write]) + "\n")

