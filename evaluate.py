# create the rollout animation of the three-player game using the log trajectory
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, markers
import os
import pandas as pd
import imageio

from utils.utils import draw_real_car, draw_real_human, plot_road_game
from scipy.spatial import Delaunay
import math
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize

from descartes import PolygonPatch

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate",       help="Things to evaluate",       choices=["train", "rollout", "spectrum"],        required=True)
parser.add_argument("--loadpath",       help="Path of experiment",       required=True)
parser.add_argument("--iteration",      help="Iteration of experiment to evaluate",     type=int)
args = parser.parse_args()

loadpath = args.loadpath

if not os.path.exists(loadpath):
    raise ValueError("Experiment does not exist")

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = points

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def spectrum():
    # check to see if there is logs folder:
    if not ("logs" in os.listdir(loadpath)):
        raise ValueError("There is no log folder in this experiment")

    # get experiment file:
    file_list = os.listdir(os.path.join(loadpath, "logs"))
    print("\t>> Found {} file(s)".format(len(file_list)))

    if len(file_list) > 1:
        index = input("Please choose which log file to use: ")
    else: 
        index = 0

    # Read log
    file_path = os.path.join(loadpath, "logs", file_list[index])
    with open(file_path, "rb") as log:
        raw_data = pickle.load(log)
    
    if args.iteration is None:
        print("\t>> Get the last iteration to render on top of spectrum")
        iteration = np.array(raw_data["xs"]).shape[0] - 1
    else:
        iteration = args.iteration

    print("\t>> Iteration to render on top: {}".format(iteration))

    max_iteration = np.array(raw_data["xs"]).shape[0]
    
    # create output folder
    output = os.path.join(loadpath, "evaluate")
    if not os.path.exists(output):
        os.makedirs(output)
    print("\t>> Output folder: " + output)

    plot_road_game()

    trajectory_spectrum_1 = None
    trajectory_spectrum_2 = None
    trajectory_spectrum_3 = None

    for i in range(max_iteration):
        data = pd.DataFrame(
            np.array(raw_data["xs"][i]).reshape((
                len(raw_data["xs"][i]), 14
            )), columns = [
                "x0", "y0", "theta0", "phi0", "vel0",
                "x1", "y1", "theta1", "phi1", "vel1",
                "x2", "y2", "theta2", "vel2"
            ]
        )

        if i == iteration:
            plt.plot(data["x0"], data["y0"], 'g', linewidth=2.0, zorder=10)
            plt.plot(data["x1"], data["y1"], 'r', linewidth=2.0, zorder=10)
            plt.plot(data["x2"], data["y2"], 'b', linewidth=2.0, zorder=10)

        if trajectory_spectrum_1 is None:
            trajectory_spectrum_1 = data[["x0", "y0"]]
        else:
            trajectory_spectrum_1 = pd.concat([trajectory_spectrum_1, data[["x0", "y0"]]])

        if trajectory_spectrum_2 is None:
            trajectory_spectrum_2 = data[["x1", "y1"]]
        else:
            trajectory_spectrum_2 = pd.concat([trajectory_spectrum_2, data[["x1", "y1"]]])
        
        if trajectory_spectrum_3 is None:
            trajectory_spectrum_3 = data[["x2", "y2"]]
        else:
            trajectory_spectrum_3 = pd.concat([trajectory_spectrum_3, data[["x2", "y2"]]])
    
    initial_state = data.iloc[0].to_numpy()
    draw_real_car(0, [initial_state])
    draw_real_car(1, [initial_state])
    draw_real_human([initial_state])

    points_1 = trajectory_spectrum_1.values
    points_2 = trajectory_spectrum_2.values
    points_3 = trajectory_spectrum_3.values

    concave_hull, edge_points = alpha_shape(points_1, 0.4)
    plt.gca().add_patch(PolygonPatch(concave_hull, fc='green', ec='green', fill=True, zorder=5, alpha=0.25))
    concave_hull, edge_points = alpha_shape(points_2, 0.4)
    plt.gca().add_patch(PolygonPatch(concave_hull, fc='red', ec='red', fill=True, zorder=5, alpha=0.25))
    concave_hull, edge_points = alpha_shape(points_3, 0.4)
    plt.gca().add_patch(PolygonPatch(concave_hull, fc='blue', ec='blue', fill=True, zorder=5, alpha=0.25))

    # create output folder
    output = os.path.join(loadpath, "evaluate")
    if not os.path.exists(output):
        os.makedirs(output)
    print("\t>> Output folder: " + output)

    plt.savefig(os.path.join(output, "spectrum.png"))

def train_process():
    folder_path = os.path.join(loadpath, "figures")

    if not os.path.exists(folder_path):
        raise ValueError("There is no such path: {}, please check again".format(folder_path))

    # Build GIF
    image_count = len([f for f in os.listdir(folder_path) if "plot-" in f])
    with imageio.get_writer('{}/evaluate_training.gif'.format(folder_path), mode='I') as writer:
        for i in range(image_count):
            filename = "plot-{}.jpg".format(i)
            image = imageio.imread(os.path.join(folder_path, filename))
            writer.append_data(image)

def final_rollout():
    # check to see if there is logs folder:
    if not ("logs" in os.listdir(loadpath)):
        raise ValueError("There is no log folder in this experiment")

    # get experiment file:
    file_list = os.listdir(os.path.join(loadpath, "logs"))
    print("\t>> Found {} file(s)".format(len(file_list)))

    if len(file_list) > 1:
        index = input("Please choose which log file to use: ")
    else: 
        index = 0

    # Read log
    file_path = os.path.join(loadpath, "logs", file_list[index])
    with open(file_path, "rb") as log:
        raw_data = pickle.load(log)

    if args.iteration is None:
        print("\t>> Get the last iteration to render")
        iteration = np.array(raw_data["xs"]).shape[0] - 1
    else:
        iteration = args.iteration

    print("\t>> Iteration to render: {}".format(iteration))

    # create output folder
    output = os.path.join(loadpath, "evaluate")
    if not os.path.exists(output):
        os.makedirs(output)
    print("\t>> Output folder: " + output)

    data = pd.DataFrame(
        np.array(raw_data["xs"][iteration]).reshape((
            len(raw_data["xs"][iteration]), 14
        )), columns = [
            "x0", "y0", "theta0", "phi0", "vel0",
            "x1", "y1", "theta1", "phi1", "vel1",
            "x2", "y2", "theta2", "vel2"
        ]
    )

    for i in range(len(data)):
        state = data.iloc[i].to_numpy()
        
        plot_road_game()
        draw_real_car(0, [state])
        draw_real_car(1, [state])
        draw_real_human([state], i%2)
        plt.pause(0.001)
        plt.savefig(os.path.join(output, 'step-{}.jpg'.format(i))) # Trying to save these plots
        plt.clf()

    # Build GIF
    image_count = len([f for f in os.listdir(output) if "step-" in f])
    with imageio.get_writer(os.path.join(output, 'evaluate_rollout.gif'), mode='I') as writer:
        try:
            for i in range(image_count):
                filename = "step-{}.jpg".format(i)
                image = imageio.imread(os.path.join(output, filename))
                writer.append_data(image)
        except FileNotFoundError:
            pass

if args.evaluate == "train":
    print("\t>> Evaluate the training process")
    train_process()
elif args.evaluate == "rollout":
    print("\t>> Evaluate the final rollout")
    final_rollout()
elif args.evaluate == "spectrum":
    print("\t>> Generate spectrum graph")
    spectrum()
else:
    raise NotImplementedError("Choose another evaluation run, current choice not supported")