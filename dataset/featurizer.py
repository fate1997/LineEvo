from rdkit import Chem
from rdkit.Chem import AllChem

from typing import List
from scipy.sparse import coo_matrix

import torch
import numpy as np
import networkx as nx

def one_hot_encoding(value: int, choices: List) -> List:
    """
    Apply one hot encoding
    :param value:
    :param choices:
    :return: A one-hot encoding for given index and length
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

Atom = Chem.rdchem.Atom
Mol = Chem.Mol

class MoleculeFeaturizer(object):
    def __init__(self, bond_features=True):
        self.bond_features = bond_features

    def _atom_featurizer(self, atom: Atom) -> List:
        # atom type one-hot encoding
        atomic_numer = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 28, 29, 
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53]
        atom_type_value = atom.GetAtomicNum()
        atom_type_features = one_hot_encoding(atom_type_value, atomic_numer)

        # atom degree one-hot encoding
        degree_choices = list(range(5))
        degree = atom.GetTotalDegree()
        degree_features = one_hot_encoding(degree, degree_choices)

        # atom hybridization one-hot encoding
        hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
        hybridization = int(atom.GetHybridization())
        hybrid_features = one_hot_encoding(hybridization, hybridization_choices)

        # chiral tag one-hot encoding
        chiral_tag_choices = list(range(len(Chem.ChiralType.names)-1))
        chiral_tag = atom.GetChiralTag()
        chiral_tag_features = one_hot_encoding(chiral_tag, chiral_tag_choices)

        # number of hydrogens one-hot encoding
        num_Hs_choices = list(range(5))
        num_Hs = atom.GetTotalNumHs()
        num_Hs_features = one_hot_encoding(num_Hs, num_Hs_choices)

        # aromatic
        aromatic_features = [1 if atom.GetIsAromatic() else 0]
        
        # concat features
        return atom_type_features + degree_features + hybrid_features + chiral_tag_features + num_Hs_features + aromatic_features
    
    def _get_postions(self, mol: Mol):
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        num_atoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        if success == -1:
            return None
        elif success == 0:
            return mol.GetConformer().GetPositions()[:num_atoms, :]
    

    def _bond_featurizer(self, bond: Chem.rdchem.Bond):
        # bond type one-hot encoding
        bond_type = bond.GetBondType()
        bond_type_one_hot_encoding = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]
        addition_bond_features = [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())]

        return bond_type_one_hot_encoding + addition_bond_features


    @property
    def feature_keys(self) -> List:
        return self.feature_dict.keys()

    def __call__(self, mol: Mol):

        self.feature_dict = {}
        # atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self._atom_featurizer(atom)) 
        self.feature_dict['x'] = atom_features

        # positions
        self.feature_dict['pos'] = self._get_postions(mol)

        # edge index and edges
        adj = Chem.GetAdjacencyMatrix(mol)
        coo_adj = coo_matrix(adj)
        self.feature_dict['edge_index'] = torch.tensor([coo_adj.row, coo_adj.col], dtype=torch.long)
        self.feature_dict['edges'] = torch.as_tensor(np.array(nx.from_edgelist(self.feature_dict['edge_index'].T.tolist()).edges))

        # edge_attr
        if self.bond_features and self.feature_dict['edges'].nelement() != 0:
            bond_features = []
            for i, j in zip(self.feature_dict['edges'][:, 0].tolist(), self.feature_dict['edges'][:, 1].tolist()):
                bond_features.append(self._bond_featurizer(mol.GetBondBetweenAtoms(i, j)))
        else:
            bond_features = []
        self.feature_dict['edge_attr'] = bond_features

        return self.feature_dict

    