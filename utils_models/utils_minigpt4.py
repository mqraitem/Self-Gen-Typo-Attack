import sys
sys.path.insert(0,'./MiniGPT')

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

from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def run_minigpt4(prompt, image, model_data, new_tokens=20):
    
    # torch.cuda.empty_cache() 

    model = model_data["model"]
    vis_processor = model_data["vis_processor"]
    conv_temp = model_data["conv_temp"]

    image = [vis_processor(image_).unsqueeze(0) for image_ in image] 
    image = torch.cat(image, dim=0)

    texts = prepare_texts(prompt, conv_temp)  
    answers = model.generate(image, texts, max_new_tokens=new_tokens)

    # print(len(answers))
    return answers