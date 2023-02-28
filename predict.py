import numpy as np

import argparse
import sys
import torch
import json

import model_func

# Optional: Allows the argparse descriptions to be displayed in a single line
# https://stackoverflow.com/questions/52605094/python-argparse-increase-space-between-parameter-and-description#comment98774195_52606755
formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)

# Create the argparser
parser = argparse.ArgumentParser(
    formatter_class=formatter,
    prog = 'Predict Flower',
    description = 'Predicts the classification of flowers',
)

# Declare parser arguments
parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', help="Path to the flower image", metavar="IMG_PATH")
parser.add_argument('checkpoint', default='./checkpoint.pth', help="Path to checkpoint.", metavar="CHKPOINT_PATH")
parser.add_argument('-k', '--top_k', type=int, default=3, help="Number of training epochs. Default = 3")
parser.add_argument('-cat', '--category_names', default='cat_to_name.json', help="Select category mapping to real names. Default = cat_to_name.json", metavar="CATEGORY")
parser.add_argument('--gpu', action='store_true', help="Use GPU (CUDA) for inference")

# If no arguments are passed then display command syntax
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

# Process the arguments data
image_file = args.input
checkpoint = args.checkpoint
top_k = args.top_k
category = args.category_names

# returns True if the flag is raised else False
use_gpu = args.gpu

# Print commands to test argparse data handling
# print(image_file)
# print(checkpoint)
# print(top_k)
# print(category)
# print(use_gpu)

# Check if --gpu flag is raised and CUDA is available
# and hence set torch device accordingly
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def prediction():
    model = model_func.load_checkpoint(checkpoint)
    with open(category, 'r') as json_file:
        name = json.load(json_file)
        
    probabilities = model_func.predict(image_file, model, device, top_k)
    probability = np.array(probabilities[0][0].cpu())
    labels = [name[str(index + 1)] for index in np.array(probabilities[1][0].cpu())]
    
    i = 0
    while i < top_k:
        print("{: >20} with a probability of {}".format(labels[i].title(), probability[i]))
        i += 1
    print("Prediction complete")

# Python idiom: Execute prediction() function only when script is directly executed from shell
if __name__== "__main__":
    prediction()