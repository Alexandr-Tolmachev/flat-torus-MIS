import numpy as np
import pandas as pd
import json
import argparse
import pathlib


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for the comparing methods found MIS")
    
    parser.add_argument("--output_folder", type=pathlib.Path, action="store", help="Folder in which the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")
    parser.add_argument("--result_filename", type=pathlib.Path, default="results.json", action="store", help="Filename of result files for different methods")
    parser.add_argument("--comparison_folder", type=pathlib.Path, action="store", help="Folder in which the comparison of various methods results should be stored")
    parser.add_argument("--sorted", type=bool, default=False, action="store", help="Sorted results via mean results over all considered methods")
    args = parser.parse_args()
    args.comparison_folder.mkdir(parents=True, exist_ok=True)

    #output_path = args.output_folder   #'data/output/'
    
    M_values = []
    N_values = []
    alpha_values = []
    l1_values = []
    l2_values = []
    mis_results_dict = {}
    
    for method_name in ['kamis', 'dgl', 'intel', 'lwd']: 
        with open(str(args.output_folder / method_name / args.result_filename), 'r') as f:  #'/results.json'
            curr_dict = json.load(f)
            
            dataset_size = len(curr_dict)
    
            M_values = [0] * dataset_size
            N_values = [0] * dataset_size
            alpha_values = [0] * dataset_size
            l1_values = [0] * dataset_size
            l2_values = [0] * dataset_size
            mis_results = [0] * dataset_size
    
            for graph_name, graph_res in curr_dict.items():
                ind = int(graph_name[6:])
                
                l1_values[ind] = graph_res['l1']
                l2_values[ind] = graph_res['l2']
                alpha_values[ind] = graph_res['alpha']
                N_values[ind] = graph_res['N']
                M_values[ind] = graph_res['M']
                
                if 'mwis_found' in graph_res.keys():
                    if graph_res['mwis_found']:
                        mis_results[ind] = graph_res['mwis_vertices']
                    else:
                        mis_results[ind] = 0
                else:
                    if graph_res['found_mis']:
                        mis_results[ind] = graph_res['vertices']
                    else:
                        mis_results[ind] = 0
            
            mis_results_dict[method_name] = mis_results
            
    results_df = pd.DataFrame({'l1': l1_values, 
                              'l2': l2_values,
                              'alpha': alpha_values,
                              'N': N_values,
                              'M': M_values})
    
    for method_name, method_results_array in mis_results_dict.items():
        results_df[method_name] = method_results_array
    
    # add column with mean results over all four methods (DGL-TreeSearch, Intel-TreeSearch, KaMIS, LwD)    
    results_df['mean'] = (results_df['kamis'] + results_df['dgl'] + results_df['intel'] + results_df['lwd']) / 4
    results_df.sort_values('mean', ascending=False, inplace=args.sorted)
    
    # save results of the comparison
    path_to_comparison_file = str(args.comparison_folder / args.result_filename).replace('.json', '.csv')
    results_df.to_csv(path_to_comparison_file)