from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
import torch
import networkx as nx
import numpy as np

from typing import List, Union
from rdkit import Chem
from EFGs import mol2frag
Mol = Chem.Mol

import numpy as np

import os
from tqdm import tqdm
import random

from .featurizer import MoleculeFeaturizer
from .feature_utils import evolve_edges_generater
import pandas as pd

class MolData(Data):
    def __init__(self, ID: str):
        super().__init__()
        self.ID = ID
        self.num_nodes = None
        self.num_bonds = None
        self.num_nodes_features = None
        self.num_bonds_features = None
        self.evolve_edges = 0
        self.flat_index = None
    
    def __repr__(self):
        return f'MolData(ID={self.ID}, num_node={self.num_nodes}, num_bond={self.num_bonds}, num_node_features={self.num_nodes_features})'

    def __inc__(self, key, value, *args, **kwargs):
        if 'edges' in key:
            return self.num_nodes if key[-1] == '0' else getattr(self, 'edges_'+str(int(key[-1])-1)).size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

 

class MolDataset(Dataset):
    def __init__(self, 
                 name: str,
                 replace: bool=False,
                 mols: List[Mol]=None,
                 labels: np.array=None,
                 task_name: List[str]=None,
                 ids: List[str]=None, 
                 featurizer: MoleculeFeaturizer=None,
                 save_folder: str='.\dataset\processed_files',
                 is_csv: bool=True):
        self.data_list = []
        self.mols = mols
        self.labels = labels
        self.task_name = task_name
        self.ids = ids
        self.featurizer = featurizer
        self.name = name
        self.processed_path = os.path.join(save_folder, self.name + '.pt' if name!='qm9' else self.name + f'_{task_name}.pt')
        
        if (replace or not os.path.exists(self.processed_path)):
            if name != 'qm9':
                if is_csv:  self.load_csv()
                self._process()
            else:
                qm9_target_dict = {
                'dipole_moment':0,
                'isotropic_polarizability':1,
                'homo': 2,
                'lumo':3,
                'gap':4,
                'electronic_spatial_extent':5,
                'zpve':6,
                'energy_U0':7,
                'energy_U':8,
                'enthalpy_H':9,
                'free_energy':10,
                'heat_capacity':11,
                }
                assert task_name in qm9_target_dict.keys()
                self.from_qm9(qm9_target_dict[task_name])
        else:
            self.data_list = torch.load(self.processed_path)
    
    def _process(self):
        
        for i, mol in enumerate(tqdm(self.mols, desc='data processing')):
            feature_dict = self.featurizer(mol)
            data = MolData(ID=self.ids[i])

            # postions
            if feature_dict['pos'] is None:
                print('Warning: No positions found for molecule {}'.format(self.ids[i]))
                continue
            data.pos = torch.tensor(feature_dict['pos'], dtype=torch.float32)

            # edge index and edges
            data.edge_index = feature_dict['edge_index']

            # edge_attr
            if 'edge_attr' in feature_dict.keys():
                data.edge_attr = torch.tensor(feature_dict['edge_attr'], dtype=torch.float32)
                data.num_bonds_features = data.edge_attr.shape[1] if data.edge_attr.shape[0] != 0 else 0

            data.y = torch.tensor(self.labels[i], dtype=torch.float32)
            data.x = torch.tensor(feature_dict['x'], dtype=torch.float32)

            # repr info
            data.num_bonds = data.edge_index.shape[1]
            data.num_nodes_features = data.x.shape[1]
            data.num_nodes = data.x.shape[0]

            self.data_list.append(data)
        
        torch.save(self.data_list, self.processed_path)
    
    def load_csv(self):
        raw_df = pd.read_csv('./dataset/raw_files/' + self.name + '.csv')
        self.featurizer = MoleculeFeaturizer()
        self.ids = raw_df.smiles.tolist()
        self.mols = []
        self.labels = raw_df.iloc[:, 1:].to_numpy()
        self.task_name = raw_df.columns[1:]

        for i, smiles in enumerate(self.ids):
            self.mols.append(Chem.MolFromSmiles(smiles))
    

    def from_qm9(self, target=0, toy=False):
        
        path = '../dataset/qm9/'
        dataset = QM9(path)
        for i in tqdm(range(len(dataset))):
            pyg_data = dataset[i]
            data = MolData(ID=pyg_data.name)
            data.pos = pyg_data.pos
            data.edge_index = pyg_data.edge_index
            data.y = pyg_data.y[:, target]
            data.x = pyg_data.x
            data.z = pyg_data.z

            # repr info
            data.num_bonds = data.edge_index.shape[1]
            data.num_nodes_features = data.x.shape[1]
            data.num_nodes = data.x.shape[0]

            self.data_list.append(data)
        
        torch.save(self.data_list, self.processed_path)

    
    def shuffle(self, seed):
        r = random.random
        random.seed(seed)
        random.shuffle(self.data_list, random=r)

    def transform(self, depth):
        
        for data in self.data_list:
            
            edges = torch.LongTensor(np.array(nx.from_edgelist(data.edge_index.T.tolist()).edges))
            
            num_nodes = data.x.shape[0]
            isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
            edges = torch.cat([edges, torch.LongTensor([[i, i] for i in isolated_nodes])], dim=0).to(torch.long)
            
            setattr(data, f'edges_{0}', edges)
            
            for i in range(depth):
                
                num_nodes = edges.shape[0]
                edges = evolve_edges_generater(edges)

                # create edges for isolated nodes
                isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
                edges = torch.cat([edges, torch.LongTensor([[i, i] for i in isolated_nodes]).to(edges.device)], dim=0)
                
                setattr(data, f'edges_{i+1}', edges)


    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)
    
    def __repr__(self):
        return f'MolDataset({self.name}, num_mols={self.__len__()})'

    
class TargetSelectTransform(object):
    def __init__(self, target_id=0):
        self.target_id = target_id

    def __call__(self, data):
        data.y = data.y[:, self.target_id]
        return data
