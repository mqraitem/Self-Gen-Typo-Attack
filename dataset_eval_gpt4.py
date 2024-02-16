import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import requests
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import os 
import random
from torch.utils.data import DataLoader
import pickle
import argparse

from datasets.dataset_pets import Pets
from datasets.dataset_cars import Cars
from datasets.dataset_flowers import Flowers
from datasets.dataset_food101 import Food101
from datasets.dataset_aircraft import Aircraft

from utils_eval.utils_eval_gpt4 import * 
from utils import get_model_method_images

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cars')
parser.add_argument('--method', type=str, default='base')
parser.add_argument('--int', type=str, default='0')

opt = parser.parse_args()

resize = 512

if opt.dataset == "pets":
    dataset = Pets("path-to-pets", "test", resize)

elif opt.dataset == "cars":
    dataset = Cars("path-to-stanford-cars", "test", resize)

elif opt.dataset == "flowers":
    dataset = Flowers("path-to-pets-oxford-flower", "valid", resize)

elif opt.dataset == "food101":
    dataset = Food101("path-to-food-101", resize)

elif opt.dataset == "aircraft":
    dataset = Aircraft("path-to-fgvc-aircraft-2013b", resize)

else: 
    raise NotImplementedError


if os.path.isfile(f"outputs/output_gpt4/{opt.dataset}_{opt.method}.txt"):
    gpt4_data_images = get_model_method_images(f"{opt.dataset}_gpt4", "gpt4")
    images_processed = gpt4_data_images["image_names_all"]
    dataset.filter_by_image_names(images_processed)

else: 
    os.makedirs(f"outputs/output_gpt4", exist_ok=True)

if opt.method == "no_text":
    run_exp_base(dataset, dataset_name = opt.dataset, image_int = opt.int)
 
#######################
# Random Class
#######################

if opt.method == "random_class":
    run_exp_base_text(dataset, opt, dataset_name = opt.dataset, text_position="top", image_int = opt.int)

if opt.method == "random_class_ignore_text":
    dataset.set_prompts(ignore_text=True)
    run_exp_base_text(dataset, opt, dataset_name = opt.dataset, text_position="top", image_int = opt.int)

#######################
# GPT4 Attacks
#######################

if opt.method == "gpt4_lvlm_attack":
    run_exp_base_text_gpt4_attack(dataset, opt, dataset_name = opt.dataset, text_position="top", describe_position="bottom", image_int = opt.int)

if opt.method == "gpt4_lvlm_attack_ignore_text":
    run_exp_base_text_gpt4_attack_llm(dataset, opt,  dataset_name = opt.dataset, text_position="top", image_int = opt.int)

if opt.method == "gpt4_llm_attack":
    dataset.set_prompts(ignore_text=True)
    run_exp_base_text_gpt4_attack(dataset, opt, dataset_name = opt.dataset, text_position="top", describe_position="bottom", image_int = opt.int)



#######################
# LLavA Attacks
#######################

if opt.method == "llava_lvlm_attack":
    run_exp_base_text_llava_attack(dataset, opt, dataset_name = opt.dataset, text_position="top", image_int = opt.int)

if opt.method == "llava_lvlm_attack_ignore_text":
    dataset.set_prompts(ignore_text=True)
    run_exp_base_text_llava_attack(dataset, opt, dataset_name = opt.dataset, text_position="top", image_int = opt.int)

#######################
# LLaMA Attacks
#######################

if opt.method == "llava_llm_attack":
    run_exp_base_text_llama_attack_llm(dataset, opt, dataset_name = opt.dataset, text_position="top", image_int = opt.int)

#######################
# CLIP Attacks
#######################

if opt.method == "llava_ve_attack":
    run_exp_base_text_clip_attack(dataset, opt, dataset_name = opt.dataset, text_position="top", describe_position="bottom", image_int = opt.int)
