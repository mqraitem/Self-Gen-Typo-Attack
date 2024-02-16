import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import requests
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import random
import string

def run_llava(prompts, images, model_data, repeat=False, new_tokens = 20): 


    prompts_fixed = [] 
    for idx, prompt in enumerate(prompts): 
        prompts_fixed.append("USER: <image>\n" + prompt + "\nASSISTANT:")

    inputs = model_data["processor"](prompts_fixed, images=images, return_tensors="pt", padding=True).to("cuda", torch.float16)


    len_inputs_ids = [len(x) for x in inputs["input_ids"]]
    output = model_data["model"].generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    output = model_data["processor"].batch_decode(output, skip_special_tokens=True)        
    output = [text.split("\nASSISTANT:")[1].lower().strip() for text in output]

    return output 