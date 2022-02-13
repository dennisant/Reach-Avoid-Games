import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir",           help="directory of experiment")
args = parser.parse_args()

if not os.path.exists(args.exp_dir):
    raise ValueError("There is no such experiment, please check again")

experiment_name = os.path.basename(os.path.normpath(args.exp_dir))
folder_name = "result/" + experiment_name + "/figures/"

if not os.path.exists(folder_name):
    raise ValueError("There is no such path: {}, please check again".format(folder_name))

# Build GIF
image_count = len(os.listdir(folder_name))
with imageio.get_writer('{}/{}.gif'.format(folder_name, experiment_name), mode='I') as writer:
    for i in range(image_count):
        filename = "plot-{}.jpg".format(i)
        image = imageio.imread(os.path.join(folder_name, filename))
        writer.append_data(image)