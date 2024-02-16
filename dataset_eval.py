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

from utils_eval.utils_eval import *
from utils import get_model_method_images

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llava')
parser.add_argument('--dataset', type=str, default='cars')
parser.add_argument('--method', type=str, default='no_text')
parser.add_argument('--color', type=str, default='black')
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


if os.path.isfile(f"outputs/output_{opt.model}/{opt.dataset}_{opt.method}.txt"):
    model_method_images = get_model_method_images(f"{opt.dataset}_{opt.method}", opt.model)
    images_processed = model_method_images["image_names_all"]
    dataset.filter_by_image_names(images_processed)

else: 
    os.makedirs(f"outputs/output_{opt.model}", exist_ok=True)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers = 2)
model_data = get_model_data(opt.model)

from utils_eval import *
import torch

#######################
# Base Case
#######################
if opt.method == "no_text":
    run_exp_base(dataloader, opt.dataset, model_data, opt)

#######################
# Random Class
#######################
if opt.method == "random_class":
    run_exp_base_text(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)

if opt.method == "random_class_ignore_text":
    dataloader.dataset.set_prompts(ignore_text=True)
    run_exp_base_text(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)


#######################
# GPT4 Attacks
#######################
if opt.method == "gpt4_lvlm_attack":
    run_exp_base_text_gpt4_attack(dataloader, opt.dataset, model_data, opt, position="top", position_desc="bottom", insert_text_check=True, insert_text_desc_check=True)

if opt.method == "gpt4_lvlm_attack_ignore_text":
    dataloader.dataset.set_prompts(ignore_text=True)
    run_exp_base_text_gpt4_attack(dataloader, opt.dataset, model_data, opt, position="top", position_desc="bottom", insert_text_check=True, insert_text_desc_check=True)

if opt.method == "gpt4_llm_attack":
    run_exp_base_text_gpt4_attack_llm(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True, insert_text_desc_check=True)

#######################
# LLaMA Attacks
#######################

if opt.method == "llava_llm_attack":
    run_exp_base_text_llama_attack_llm(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)

#######################
# LLavA Attacks
#######################

if opt.method == "llava_lvlm_attack":
    run_exp_base_text_llava_attack(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)

if opt.method == "llava_lvlm_attack_ignore_text":
    dataloader.dataset.set_prompts(ignore_text=True)
    run_exp_base_text_llava_attack(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)


#######################
# CLIP Attacks
#######################

if opt.method == "llava_ve_attack":
    run_exp_base_text_clip_attack(dataloader, opt.dataset, model_data, opt, position="top", insert_text_check=True)
