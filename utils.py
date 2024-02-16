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
import transformers


def get_gpt4_data(dataset): 
	data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_gpt4_detail/{dataset}.txt", sep= ",", header=None)
	wrong_labels_new_all = data_attacks[2].tolist()
	wrong_labels_new_all = [x.strip() for x in wrong_labels_new_all]
	descriptions_all = data_attacks[1].tolist()
	descriptions_all = [x.strip() for x in descriptions_all]
	image_names_all = data_attacks[0].tolist()

	return { 
		"wrong_labels_new_all":wrong_labels_new_all,
		"descriptions_all":descriptions_all,
		"image_names_all":image_names_all
	}

def get_random_data(dataset): 
	data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_random/{dataset}.txt", sep= ",", header=None)
	wrong_labels_new_all = data_attacks[1].tolist()
	wrong_labels_new_all = [x.strip() for x in wrong_labels_new_all]
	image_names_all = data_attacks[0].tolist()

	return { 
		"wrong_labels_new_all":wrong_labels_new_all,
		"image_names_all":image_names_all
	}

def get_gpt4_data_llm(dataset):
	if os.path.exists(f"outputs/outputs_attacks/output_gpt4_detail_llm/{dataset}.txt"): 
		data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_gpt4_detail_llm/{dataset}.txt", sep= ",", header=None)
		gt_label = data_attacks[0].tolist()
		description_all = data_attacks[1].tolist()
		wrong_label_new = data_attacks[2].tolist()

		gt_label = [x.strip().lower() for x in gt_label]
		description_all = [x.strip().lower() for x in description_all]
		wrong_label_new = [x.strip().lower() for x in wrong_label_new]

		return {
			"gt_labels":gt_label,
			"descriptions_all":description_all,
			"wrong_labels_new_all":wrong_label_new
		}
	else: 
		return {
			"gt_labels":[],
			"descriptions_all":[],
			"wrong_labels_new_all":[]
	}

def get_llama_data_llm(dataset): 
	if os.path.exists(f"outputs/outputs_attacks/output_llama_detail_llm/{dataset}.txt"):
		data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_llama_detail_llm/{dataset}.txt", sep= ",", header=None)
		gt_label = data_attacks[0].tolist()
		wrong_label_new = data_attacks[1].tolist()

		gt_label = [x.strip().lower() for x in gt_label]
		wrong_label_new = [x.strip().lower() for x in wrong_label_new]

		return {
			"gt_labels":gt_label,
			"wrong_labels_new_all":wrong_label_new
		}
	else: 
		return {
			"gt_labels":[],
			"wrong_labels_new_all":[]
		}
		
def get_clip_data(dataset): 
	if os.path.exists(f"outputs/outputs_attacks/output_clip/{dataset}.txt"):
		data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_clip/{dataset}.txt", sep= ",", header=None)
		image_names = data_attacks[0].tolist()
		wrong_label_new = data_attacks[1].tolist()

		image_names = [x.strip() for x in image_names]
		wrong_label_new = [x.strip().lower() for x in wrong_label_new]

		return {
			"image_names":image_names,
			"wrong_labels_new_all":wrong_label_new
		}
	else: 
		return {
			"image_names":[],
			"wrong_labels_new_all":[]
		}

def get_llava_data(dataset): 
	if os.path.exists(f"outputs/outputs_attacks/output_llava_detail/{dataset}.txt"):
		data_attacks = pd.read_csv(f"outputs/outputs_attacks/output_llava_detail/{dataset}.txt", sep= ",", header=None)
		image_names = data_attacks[0].tolist()
		wrong_label_new = data_attacks[1].tolist()

		image_names = [x.strip() for x in image_names]
		wrong_label_new = [x.strip().lower() for x in wrong_label_new]

		return {
			"image_names":image_names,
			"wrong_labels_new_all":wrong_label_new
		}
	else: 
		return {
			"image_names":[],
			"wrong_labels_new_all":[]
		}


def get_model_method_images(file_name, model): 
	data_attacks = pd.read_csv(f"outputs/output_{model}/{file_name}.txt", sep= ",", header=None)
	image_names_all = data_attacks[0].tolist()
	return {
		"image_names_all":image_names_all
	}


def add_white_background(image): 
	image_width, image_height = image.size
	white_height = int(image_height * 0.25)
	new_image = Image.new('RGB', (image_width, image_height + white_height*2), 'white')
	new_image.paste(image, (0, white_height))
	return new_image, image_height

def insert_text(image, text, image_height_old, font_path = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", font_size = 20, color = "black", position="top"): 

	image_width, image_height = image.size

	# Set up the font
	try:
		font = ImageFont.truetype(font_path, font_size)
	except IOError:
		font = ImageFont.load_default()

	# Prepare text wrapping
	lines = textwrap.wrap(text, width=int(image_width / (font_size * 0.45)))

	# Initialize drawing context
	draw = ImageDraw.Draw(image)

	# Write text
	# y_text = image_height_old
	y_text = 0 if position == "top" else image_height_old + int(image_height_old * 0.25)
	for line in lines:
		line_width, line_height = font.getsize(line)
		draw.text((0, y_text), line, font=font, fill=color)
		y_text += line_height + 5

	return image

def get_model_data(name): 
	if name == "blip":

		from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
		from utils_models.utils_instructblip import run_blip 

				
		processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
		model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl",  torch_dtype=torch.float16).cuda()


		model_data = {
			"model":model, 
			"processor":processor,
			"name":"blip"
		}

	if name == "llava":

		from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline

		model_id = "llava-hf/llava-1.5-13b-hf"

		model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
		processor = AutoProcessor.from_pretrained(model_id)
		processor.image_processor.do_center_cropq = False
		processor.image_processor.size = {"height": 336, "width": 336}

		model_data = {
			"model":model, 
			"processor":processor,
			"name":"llava"
		}

	if name == "minigpt4": 

		import sys
		sys.path.insert(0,'./models/MiniGPT')

		from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
		from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
		from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

		from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
		from minigpt4.conversation.conversation import CONV_VISION_minigptv2
		from minigpt4.common.config import Config

		from utils_models.utils_minigpt4 import run_minigpt4 

		def list_of_str(arg):
			return list(map(str, arg.split(',')))

		class Args: 
			def __init__(self): 
				self.cfg_path = "path-to-minigpt4/MiniGPT/eval_configs/minigptv2_eval.yaml"
				self.name = 'A2'
				self.eval_opt = 'all'
				self.max_new_tokens = 10
				self.batch_size = 32
				self.lora_r = 64
				self.lora_alpha = 16 
				self.options=None

		args = Args() 
		cfg = Config(args)

		model, vis_processor = init_model(args)
		conv_temp = CONV_VISION_minigptv2.copy()
		conv_temp.system = ""
		model.eval()
		# save_path = cfg.run_cfg.save_path

		model_data = { 
			"model": model,  
			"name": "minigpt4", 
			"conv_temp":conv_temp,
			"vis_processor":vis_processor,
		}

	if name == "clip":
		import open_clip
		model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')	
		tokenizer = open_clip.get_tokenizer('ViT-B-32')

		model_data = {
			"model":model, 
			"preprocess":preprocess,
			"tokenizer":tokenizer,
			"name":"clip"
		}

	return model_data 

def eval_output(out, answer): 
	answer = answer.lower() 
	out = out.lower()

	if answer in out: 
		return 1 

	else: 
		return 0 
