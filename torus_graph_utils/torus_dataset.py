from torch.utils.data import Dataset
from torus_graph_utils.torus_graph import is_perfectly_periodical
import numpy as np


class TorusGraphDataset(Dataset):
    """Torus graph dataset."""

    def __init__(self):
        self.data = {'l1': [], 'l2': [], 'alpha': [], 'N': [], 'M': []}
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert idx < self.length and idx >= 0, "Index out of range!"

        curr_item = (self.data['l1'][idx], 
                     self.data['l2'][idx], 
                     self.data['alpha'][idx],
                     self.data['N'][idx], 
                     self.data['M'][idx])
        
        return curr_item

    def add(self, torus_dict):
        for name, value in torus_dict.items():
            self.data[name].append(value)
        self.length += 1
        
    
     
def create_dataset_for_experiment(l1_range=(2, 6), l1_step=0.2, 
                                  l2_range=(2, 6), l2_step=0.2, 
                                  alpha_range=(20, 90), alpha_step=15, 
                                  N=100, M=100):
    '''
    This fuction creates the dataset based on flat torus graphs from the defined range 
    with the same grid parameters (N, M)
    '''
    l1_values = np.array([l1_range[0] + t * l1_step for t in range(int(l1_range[1] / l1_step) + 1) 
                          if l1_range[0] + t * l1_step <= l1_range[1]], dtype=np.float64)
    l2_values = np.array([l2_range[0] + t * l2_step for t in range(int(l2_range[1] / l2_step) + 1) 
                          if l2_range[0] + t * l2_step <= l2_range[1]], dtype=np.float64)     
    alpha_values = np.array([alpha_range[0] + t * alpha_step for t in range(int(alpha_range[1] / alpha_step) + 1) 
                             if alpha_range[0] + t * alpha_step <= alpha_range[1]], dtype=np.float64)

    print('l1_grid: ', l1_values)
    print('l2_grid: ', l2_values)
    print('alpha_grid:', alpha_values)
    
    alpha_values *= (np.pi / 180)  # from dergees to radians, for instance: 90 degrees equals to pi / 2
    
    dataset = TorusGraphDataset()  # test dataset
    
    for l1 in l1_values:
        for l2 in l2_values:
            for alpha in alpha_values:
                if l1 <= l2 and is_perfectly_periodical(l1, l2, alpha): # add only perfectly periodical flat toruses
                    dataset.add({'l1': l1, 'l2': l2, 'alpha': alpha, 'N': N, 'M': M})
                    
    return dataset


def create_dataset_for_one_graph(l1, l2, alpha, N_grid=[100], M_grid=[100]):
    '''
    Create dataset for one graph and different grid (N, M) sizes
    '''
    dataset = TorusGraphDataset()
    
    alpha *= (np.pi / 180) # from degrees to radians
    
    # add only perfectly periodical flat toruses
    assert is_perfectly_periodical(l1, l2, alpha), "Torus is not a perfectly periodical"
        
    for N, M in zip(N_grid, M_grid):
        dataset.add({'l1': l1, 'l2': l2, 'alpha': alpha, 'N': N, 'M': N})
    
    return dataset