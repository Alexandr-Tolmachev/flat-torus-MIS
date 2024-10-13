import argparse
import pathlib
import logzero
from logzero import logger
from filelock import FileLock

from main import main  # import main function
#from torus_dataset import TorusGraphDataset
from torus_graph_utils.torus_dataset import create_dataset_for_experiment, create_dataset_for_one_graph
from torus_graph_utils.torus_graph import get_torus_graph

import numpy as np
import networkx as nx
import shutil
import os
import json

# globals for release in the end
cuda_devices = []
got_devices_from_folder = False 
    

def _release_cuda_devices(cuda_devices, path: pathlib.Path):
    for cuda_device in cuda_devices:
        gpu_file = path / (str(cuda_device) + ".gpu")
        gpu_file.touch(exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testbench for MIS solvers.")
    subparsers = parser.add_subparsers(help='sub-command help', dest="operation")

    # Global flags
    parser.add_argument("--loglevel", type=str, action="store", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Verbosity of logging (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--self_loops", action="store_true", default=False, help="Enable self loops addition (in input data) for GCN-based model.")

    
    solve_parser = subparsers.add_parser("solve", help="Call a solver")
    solve_parser.add_argument("--time_limit", type=int, nargs="?", action="store", default=600, help="Time limit in seconds")
    solve_parser.add_argument("--num_threads", type=int, nargs="?", action="store", default=8, help="Maximum number of threads to use.")
    solve_parser.add_argument("--weighted", action="store_true", default=False, help="If enabled, solve the weighted MIS problem instead of MIS.")
    solve_parser.add_argument("--cuda_devices", type=int, nargs="*", action="store", default=[], help="Which cuda devices should be used (distributed around the threads in round-robin fashion). If not given, CUDA is disabled.")
    solve_parser.add_argument("--num_cuda_devices", type=int, action="store", default=0, help="Alternative to --cuda_devices. Uses a folder to manage available GPUs.")
    solve_parser.add_argument("--cuda_device_folder", type=pathlib.Path, action="store", default="/tmp/gpus", help="Folder containing a lockfile for the GPU management. ")

    solve_parser.add_argument("--pretrained_weights", type=pathlib.Path, nargs="?", action="store", help="Pre-trained weights to be used for solving/continuing training.")
    solve_parser.add_argument("--reduction", action="store_true", default=False, help="If enabled, reduce graph during tree search.")
    solve_parser.add_argument("--local_search", action="store_true", default=False, help="If enabled, use local_search if time left.")
    solve_parser.add_argument("--queue_pruning", action="store_true", default=False, help="(DGL-Treesearch only) If enabled, prune search queue.")
    solve_parser.add_argument("--noise_as_prob_maps", action="store_true", default=False, help="(DGL-Treesearch and LwD only) If enabled, use uniform noise instead of GNN output.")
    solve_parser.add_argument("--weighted_queue_pop", action="store_true", default=False, help="(DGL-Treesearch only) If enabled, choose element from queue with probability inverse proportional to # of unlabelled vertices in it.")
    solve_parser.add_argument("--max_prob_maps", type=int, action="store", help="DGL-TS specific: number of probability maps to use.")
    solve_parser.add_argument("--model_prob_maps", type=int, action="store", help="Treesearch (Intel/DGL) specific: Number of probability maps the model was/should be trained for.")
    solve_parser.add_argument("--maximum_iterations_per_episode", type=int, action="store", help="LwD specific: Maximum iterations before the MDP timeouts.")
    solve_parser.add_argument("--max_nodes", type=int, action="store", help="LwD specific: If you have lots of graphs, the determiniation of maximum number of nodes takes some time. If this value is given, you can force-overwrite it to save time.")
    solve_parser.add_argument("--quadratic", action="store_true", default=False, help="Gurobi specific: Whether a quadratic program should be used instead of a linear program to solve the MIS problem (cannot be used together with weighted)")
    solve_parser.add_argument("--write_mps", action="store_true", default=False, help="Gurobi specific: Instead of solving, write mps output (e.g., for tuning)")
    solve_parser.add_argument("--prm_file", type=pathlib.Path, nargs="?", action="store", help="Gurboi specific: Gurobi parameter file (e.g. by grbtune).")

    solve_parser.add_argument("solver", type=str, help="Solver to use.", choices=["dgl-treesearch", "intel-treesearch", "gurobi", "kamis", "lwd"])
    solve_parser.add_argument("input_folder", type=pathlib.Path, action="store", help="Directory containing input")
    solve_parser.add_argument("output_folder", type=pathlib.Path, action="store",  help="Folder in which the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")

    ## ADDED ARGUMENTS
    solve_parser.add_argument("--without_result_overwrite", action="store_true", default=False, help="Without overwrite the result file.")
    
    # argument for the fast reproducing experiments from our paper
    solve_parser.add_argument("--dataset_name", type=str, action="store", default=None, help="Name of the dataset used in the experiment")
    solve_parser.add_argument("--path_to_datasets_params", type=pathlib.Path, nargs="?", action="store", help="Path to datasets params")
    
    # arguments for one graph datasets
    solve_parser.add_argument("--l1", type=float, action="store", default=3.331, help="l1 value")
    solve_parser.add_argument("--l2", type=float, action="store", default=3.331, help="l2 value")
    solve_parser.add_argument("--alpha", type=int, action="store", default=60, help="alpha angle value in degrees")
    solve_parser.add_argument("--N_values", type=int, nargs="*", action="store", default=[100], help="N values")
    solve_parser.add_argument("--M_values", type=int, nargs="*", action="store", default=[100], help="M values")
    
    # result filename argument
    solve_parser.add_argument("--results_filename", type=str, action="store", default='curr_results.json', help="Results json filename, default: curr_results.json")
  
    # Optionals without default (defaults are set in the model)
    args = parser.parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True) # directory to save results
    args.input_folder.mkdir(parents=True, exist_ok=True) # directory for created graphs

    try:
        assert args.results_filename != 'results.json', 'Wrong result filename! Please choose another filename: for instance, results0.json'
        
        if args.dataset_name is not None:
            # run experiments on the some dataset from the paper if the dataset name is provided
            dataset_name = args.dataset_name 
            
            # download parameters of datasets that have used in experiments
            with open(args.path_to_datasets_params) as f:
                exp_dataset_params = json.load(f)
            
            dataset = create_dataset_for_experiment(l1_range=exp_dataset_params[dataset_name]['l1_range'],
                                                    l1_step=exp_dataset_params[dataset_name]['l1_step'],
                                                    l2_range=exp_dataset_params[dataset_name]['l2_range'],
                                                    l2_step=exp_dataset_params[dataset_name]['l2_step'],
                                                    alpha_range=exp_dataset_params[dataset_name]['alpha_range'],
                                                    alpha_step=exp_dataset_params[dataset_name]['alpha_step'])
        else:
            # run experiments 
            print(args.N_values, args.M_values)
            # create dataset based on parameters transmitted as arguments
            dataset = create_dataset_for_one_graph(args.l1, args.l2, args.alpha, args.N_values, args.M_values)

        
        print(f'Dataset of size {len(dataset)} has been created!')
        
        shutil.rmtree(args.input_folder) # clear input folder before starting processing
        os.makedirs(args.input_folder)
        
        path_to_json = str(args.output_folder / 'results.json')
        
        try:
            os.remove(pathlib.Path(path_to_json)) # delete results.json file if it was in output directory
        except OSError:
            pass
        
        for i in range(len(dataset)):
            
            graph = get_torus_graph(*dataset[i])
            
            path_to_input_graph_save = str(args.input_folder / f"graph_{i}.gpickle")
        
            nx.write_gpickle(graph, path_to_input_graph_save)
            del graph 
            
            print(f"Graph {i+1} of {len(dataset)} created!")
        
            args.without_result_overwrite = True # add results for graph at the end of results json file
            main(args)
        
            print(f"{i + 1}/{len(dataset)} graphs considered!")
            
            shutil.rmtree(args.input_folder) # the graph wiil be deleted after its processing to reduce the memory consumption
            os.makedirs(args.input_folder)
            
        print('Finish! Next the json file will be created...')
        
        
        # slightly modify the json file
        with open(path_to_json, 'r') as f:
            data = f.read()
            data = data.replace('}{', ',')
  
        with open(path_to_json, 'w') as f:
            f.write(data)
            
        # save torus graph parameters to json
        with open(path_to_json) as f:
            results = json.load(f)
            
        for graph_name, graph_results in results.items():
            graph_index = int(graph_name[graph_name.find('_') + 1:])
            l1, l2, alpha, N, M = dataset[graph_index]
            
            results[graph_name]['l1'] = l1
            results[graph_name]['l2'] = l2
            results[graph_name]['alpha'] = alpha
            results[graph_name]['N'] = N
            results[graph_name]['M'] = M
            
        
        pathlib.Path(path_to_json).unlink() # delete temporary json
        
        # create final json with defined filename
        path_to_final_json = str(args.output_folder / args.results_filename)
        
        with open(path_to_final_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)
        
            
        #with open(path_to_json, 'w', encoding='utf-8') as f:
            #json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)
            
        
        print('Results JSON created!')

        
    except KeyboardInterrupt:
        pass
    finally:
        if got_devices_from_folder:
            logger.info("Releasing cuda devices.")
            _release_cuda_devices(cuda_devices, args.cuda_device_folder)