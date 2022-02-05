import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

experiment_name = "one_player_time_inconsistent_2022-02-04-22_53"
folder_name = "result/" + experiment_name + "/figures/"
# Build GIF
image_count = len(os.listdir(folder_name))
with imageio.get_writer('{}/{}.gif'.format(folder_name, experiment_name), mode='I') as writer:
    for i in range(image_count):
        filename = "plot-{}.jpg".format(i)
        image = imageio.imread(os.path.join(folder_name, filename))
        writer.append_data(image)