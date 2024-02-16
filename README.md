# Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks 

Official implementation of [Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks](https://arxiv.org/abs/2402.00626). 

## TODO
- [ ] Add Attack Generation Code.
- [X] Add Eval Code. 

## Setup

### Prepare dataset.

- StanfordCars  
Download [StanfordCars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) dataset 

- Aircraft  
Download [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) 

- OxfordPets  
Download [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/)

- Food101  
Download [Food101](https://www.kaggle.com/datasets/dansbecker/food-101)

- Flowers  
Download [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

For each dataset, set the right path at dataset_eval.py and dataset_eval_gpt4.py

### Model Setup 

To setup 

- LLaVA/InstructBLIP, simply make sure you have transformers installed. 
- MiniGPT4, make sure to clone their repo, follow their instructions, and then set up the path to the config file in utils.py line 236. 

## Eval.

To evaluate LLaVA/InstructBlip/MiniGPT-4, run: 

```
python dataset_eval.py --model [llava/blip/minigpt4] --method [Method] --dataset [Dataset]
```

To evaluate GPT-4, first set your api key at utils_models/utils_gpt4.py, and then run: 

```
python dataset_eval_gpt4.py --method [Method] --dataset [Dataset]
```

## Citation 

If you find this repository useful please give it a star and cite as follows! :) :
```
    @article{qraitem2024vision,
    title={Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks},
    author={Qraitem, Maan and Tasnim, Nazia and Saenko, Kate and Plummer, Bryan A},
    journal={arXiv preprint arXiv:2402.00626},
    year={2024}
    }
```
