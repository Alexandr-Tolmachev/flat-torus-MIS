import numpy as np
import pandas as pd
import json
import argparse
import pathlib

import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for the comparing methods found MIS")
    
    parser.add_argument("--path_to_json", type=pathlib.Path, action="store", help="Path to JSON where the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")
    parser.add_argument("--image_folder", type=pathlib.Path, action="store", help="Path to the folder with the constructed images of the found independent sets on the flat torus.")
    parser.add_argument("--point_size", type=int, default=5, action="store", help="Point size related constant. Default values equals 20.")
    args = parser.parse_args()
    args.image_folder.mkdir(parents=True, exist_ok=True)
    
    with open(args.path_to_json, 'r') as f: 
        curr_dict = json.load(f)
        
    
    for graph_name, graph_dict in curr_dict.items():
        
        l1 = graph_dict['l1']
        l2 = graph_dict['l2']
        alpha = graph_dict['alpha']
        N = graph_dict['N']
        M = graph_dict['M']
        mis_vertices = np.array(graph_dict['mis' if 'mis' in graph_dict.keys() else 'mwis'])
        
        v1 = np.array([l1, 0])
        v2 = np.array([l2 * np.cos(alpha), l2 * np.sin(alpha)])
    
        parallelogram_coords = np.vstack([np.zeros(2), v1, v1 + v2, v2, np.zeros(2)])
    
        ind_set_coords = v1 * (mis_vertices[:, None] // M) / N + v2 * (mis_vertices[:, None] % M) / M
    
        C = 20 / (v1[0] + v2[0])  # calibrate the point size on the picture
        plt.figure(figsize=(20, C * (v1[1] + v2[1])))
        plt.plot(parallelogram_coords[:, 0], parallelogram_coords[:, 1], color='black')
        plt.scatter(ind_set_coords[:, 0], ind_set_coords[:, 1], s=int(args.point_size * 100 / N), color='black')
        plt.axis('equal')
        plt.axis('off')
        
        
        # save image
        path_to_curr_image = str(args.image_folder / graph_name) + f'_{N}x{M}_{len(mis_vertices)}.png'
        plt.savefig(path_to_curr_image, bbox_inches='tight')
        plt.show()