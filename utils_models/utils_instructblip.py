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
import random
import string


def run_blip(prompt, images, model_data, new_tokens = 20):

	inputs = model_data["processor"](images, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device="cuda", dtype=torch.float16)
		
	outputs = model_data["model"].generate(
		**inputs,
		do_sample=False,
		max_new_tokens=new_tokens,
	).detach().cpu()

	text_output = model_data["processor"].batch_decode(outputs, skip_special_tokens=True)
	text_output = [text.strip().lower() for text in text_output]
	
	return text_output 