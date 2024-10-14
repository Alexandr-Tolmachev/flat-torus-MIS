<div align="center"><h1>&nbsp;Density estimation of planar sets without unit
distances via periodic colorings exploration</h1></div>

<p align="center">
| <a href="https://www.overleaf.com/read/vwccvrvmxbck#01f7d9"><b>Paper</b></a> |


---

## Introduction 
We study a periodic planar colorings as the completely new view on the estimation of this density and empirically
confirmed current lower bound over periodic colorings. This repository based on the [paper](https://www.overleaf.com/read/vwccvrvmxbck#01f7d9) that proposed to estimate the density of planar sets without unit distances via the finding the Maximum Independent Sets (MIS) of some graphs constructed on the flat torus $T_{l_1, l_2, \alpha}$ parametrized lengths of torus sides $l_1, l_2$ and the angle $\alpha \in (0, \pi/2]$ between them.

For the benchmarking the MIS finding, the open-source benchmark suite [MWIS-Benchmark](https://github.com/MaxiBoether/mis-benchmark-framework) for the NP-hard Maximum Independent Set have been applied and integrated to the pipeline of experiments described in the our paper. Four maximum independent set solvers from this benchmark have been considered: 

- DGL-TreeSearch
- KaMIS
- Intel-TreeSearch
- Learning What to Differ

## Contents
- [Introduction](#introduction)
- [Contents](#contents)
- [Repository Contents](#repository-contents)
  - [Data and models](#data-and-models)
- [Installation](#installation)
  - [Install From The Source](#install-from-the-source)
- [Experiments](#experiments)
  - [Inference for various methods](#inference-for-various-methods)
  - [Experiments with graph datasets](#experiments-with-graph-datasets)
    - [Running on pre-defined datasets](#running-on-pre-defined-datasets)
    - [Running on the flat torus with different scales](#running-on-the-flat-torus-with-different-scales)
    - [MIS solvers comparison](#mis-solvers-comparison)
    - [Images of MIS on the flat toruses](#images-of-mis-on-the-flat-toruses)
    - [Image creation for graphs with local optimum parameters](#image-creation-for-graphs-with-local-optimum-parameters)
- [Citation](#citation)


## Repository Contents

In `solvers`, you can find the wrappers for the currently supported solvers. In the `scripts` folder, you find some scripts that could be helpful when doing analyses with this suite: 

- aggregating results for various MIS solvers
- creating images with found independent set on the flat toruses

For using this suite, `conda` is required. You can the `setup_bm_env.sh` script which will setup the conda environment with all required dependencies. The file `main.py` is the main interface you will call for data generation, solving, and training from the [MIS-benchmark](https://github.com/MaxiBoether/mis-benchmark-framework/blob/master/main.py) that we used. You can find out more about the usage using `python main.py -h`. 

File `utils.py` сontains necessary functions from the original MWIS-Benchmark.

The folder `torus_graph_utils` contains several scripts for constructing graphs based on the flat toruses and for the creation of datasets with graphs parameters that will be used in our experiments.

`solve_torus_dataset.py` script is **the key tool for running experiments**. The json file `dataset_params.json` contains the parameters of datasets that precisely described and considered in the our paper.

### Data and models

The archive `pretrained_models.zip` consists of the weights of pretrained models used in the original benchmark during their experiments ([paper](https://openreview.net/pdf?id=mk0HzdqY7i1), ICLR-2022). Before you starting experiments, you should unzip extract these files to the `pretrained_models` folder. Additionally, the Intel tree search model that was trained by Li et al. can be downloaded from [the original repository](https://github.com/isl-org/NPHard/tree/master/model).

The archieve `data.zip` contains the folder `data` contains our the experiment results (json files, MIS images on the torus that was obtained and described in [our paper](https://www.overleaf.com/read/vwccvrvmxbck#01f7d9). You should download this archieve to the root of the current repository.

```
├── flat-torus-MIS
  ├── solvers
  ├── scripts
  ├── torus_graph_utils
  ├── *pretrained_models*
  ├── *data*
  ├── solve_torus_dataset.py
  ...
```

You can download these zip-archievs from this [link](https://disk.yandex.ru/d/Es4ungd-cE0SZw). 


## Installation

Firstly, you must install `conda` and clone this repository, unzip `pretrained_models.zip` archive to `pretrained_models` folder and unzip `data.zip` to `data` folder inside this repository. 

These archieves (`data.zip` and `pretrained_models.zip`) can be downloaded via this [link](https://disk.yandex.ru/d/Es4ungd-cE0SZw) and extracted to the `data` and `pretrained_models` folders in this repository.

Next, you can run `setup_bm_env.sh` script or run two commands manually:
- `conda env create -f environment.yml` (create the environment via `environment.yml` file from this repository)
-  `conda activate mwis-benchmark` (activate the created environment)

### Install From The Source
```bash
git clone https://github.com/Alexandr-Tolmachev/flat-torus-MIS.git
cd flat-torus-MIS
unzip pretrained_models.zip
unzip data.zip
./setup_bm_env.sh
```

## Experiments

We will use the `solve_torus_dataset.py` script to run experiments. The argument `--result_filename` corresponds to the name of file with experiments results. For usage-friendliness notice that **this argument shouldn't be equal  to `results.json`** because this may result in overwriting your results from previous experiments.

### Inference for various methods
The basic example (with defaults settings) run this methods over the local maximum graph $T_{l_1, l_2, \alpha}$ found in our paper ($l_1 = l_2 = 3.331, \alpha = 60^\circ$) with grid size $N = M = 100$ (basic grid size for our experiments).

- DGL-TreeSearch
  ```python solve_torus_dataset.py --self_loops solve dgl-treesearch data/input/dgl data/output/dgl --time_limit 100 --pretrained_weights pretrained_models/dgl-final-model/1630454124_final_model32.torch --max_prob_maps 16 --num_threads 1 --reduction --local_search --queue_pruning --weighted_queue_pop --cuda_devices 0 --results_filename example_results.json```
  
- KaMIS
  ```python solve_torus_dataset.py solve kamis  data/input/kamis  data/output/kamis --time_limit 100 --results_filename example_results.json```

- Intel-TreeSearch
  ```python solve_torus_dataset.py --self_loops solve intel-treesearch  data/input/intel  data/output/intel --time_limit 100 --pretrained_weights solvers/intel_treesearch/NPHard/model --reduction --local_search  --num_threads 1 --results_filename example_results.json```

- Learning What to Differ
  ```python solve_torus_dataset.py --self_loops solve lwd data/input/lwd  data/output/lwd --time_limit 100 --pretrained_weights pretrained_models/lwd-final-model --maximum_iterations_per_episode 100 --results_filename example_results.json```

*Notice:* for Intel-TreeSearch you can use other model `pretrained_models/intel-final-model/model` 9from MWIS-Benchmark), but in our experiments the model pretrained by [Li et.al.(2019)](https://proceedings.neurips.cc/paper/2018/file/8d3bba7425e7c98c50f52ca1b52d3735-Paper.pdf) was used (pretrained model weights are located here: *solvers/intel_treesearch/NPHard/model*).

How to run experiments over graph datasets we will described in the next section.

### Experiments with graph datasets

#### Running on pre-defined datasets
The file `datasets_params.json` contains the parameters of graph_datasets. Optional arguments `--dataset_name` and `--path_to_datasets_params` related with this case. Below you can find the example of the inference on the dataset named `test_dataset` (small dataset with 6 flat torus based different graphs) from this file:

```python solve_torus_dataset.py solve kamis  data/input/kamis  data/output/kamis --time_limit 100 --results_filename example_results.json --dataset_name test_dataset --path_to_datasets_params datasets_params.json --results_filename example_test_dataset_results.json```

#### Running on the flat torus with different scales

The created intefrace allows to run experiments on the manually created datasets over one graph with various grid scales. Optional arguments `--l1`, `--l2`, `--alpha` requires the parameters of the graph $T_{l_1, l_2, \alpha}$ and arguments `N_values` and `M_values` requires the equal number of grid scales: $[n_1, n_2, \dots, n_k]$ and $[m_1, m_2, \dots, m_k]$. After that the script run the inference on the dataset consisting of $k$ graphs constructed on grids with sizes $[n_i, m_i]$ respectively.

To run KaMIS on two graphs with grids $50x50$ and $100x100$ based on the flat torus $T_{4, 4, \pi/2}$ you can run such command:
```python solve_torus_dataset.py solve kamis  data/input/kamis  data/output/kamis --time_limit 100 --results_filename example_graph_with_different_scales_results.json --l1 4 --l2 4 --alpha 90 --N_values 50 100 --M_values 50 100```

### MIS solvers comparison

To compare MIS solvers on some dataset you firstly should run experiments over this dataset (see previous subsection) with *same results_filename* (json-file) and following folder scructure:
```
├── path_to_output_folder
  ├── dgl
    ├── results_filename
  ├── kamis
    ├── results_filename
  ├── intel
    ├── results_filename
  ├── lwd
    ├── results_filename
```

After it you should run the `compare_methods.py` script from the `scripts` folder via following command to save the csv-file with same name as results_filename in the `path_to_comparison_file` directory:

```python scripts/compare_methods.py --output_folder path_to_output_folder --result_filename results_filename --comparison_folder path_to_comparison_file```

 Optional boolean argument `--sorted` corresponds the sorting the dataset results in descending order over `mean` MIS size over four considered methods.

For example, to compare the results over `dataset-1` this command has been run:

```python scripts/compare_methods.py --output_folder data/output --result_filename results_dataset-1.json --comparison_folder data/methods_comparison```

The folder `data/methods_comparision` contains of the csv-files corresponded to datasets considered in the paper.

### Images of MIS on the flat toruses

The script `create_images.py` from `scripts` folder helps to create images of MIS size on the flat torus via the json-file with results. The argument `--path_to_json` corresponds to the json file with results (including found MIS set), the argument `--image_folder` be the path where the created image will be saved. The optional argument `--point_size` corresponded to the size of each point from MIS and selected empirically. Each image saves in the format `{graph_name}_{N}_{M}_{MIS set size}.png`.

The example of usage of this script:

```python scripts/create_images.py --path_to_json data/output/kamis/example_results.json --image_folder data/images/kamis/example```

### Image creation for graphs with local optimum parameters

Consider the example of the applying of KaMIS approach to the graph with local maximum mean (over four considered methods) independent set size. As we obtain in our experiments: $l_1^* = l_2^* = 3.331, \alpha^* = 60^\circ$.

Firstly, run KaMIS on the graph with these parameters and various grid sizes: 100x100, 200x200, 300x300, 400x400:

```python solve_torus_dataset.py solve kamis  data/input/kamis  data/output/kamis --time_limit 100 --results_filename local_max_graph_results.json --l1 3.331 --l2 3.331 --alpha 60 --N_values 100 200 300 --M_values 100 200 300```

Next, create images with found MIS and save it to the corresponding folder with appropriate filenames:

```python scripts/create_images.py --path_to_json data/output/kamis/local_max_graph_results.json --image_folder data/images/kamis/local_max_graph```

For other methods (DGL-TreeSearch, Intel-TreeSearch, LwD) this procedure are analogously (the choice of the maximal grid size for each method based on RAM memory rescrictions).

## Citation
```bibtex
@misc{flat_torus_MIS2024,
      title={Density estimation of planar sets without unit distances via periodic colorings exploration}, 
      author={Alexander Tolmachev},
      year={2024},
      eprint={appears later :) },
      archivePrefix={arXiv},
      primaryClass={math.MG}
}
```
