from typing import List
import torch

from collections import defaultdict
from itertools import combinations, chain


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
        
 
def evolve_edges_generater(edges):
    l = edges[:, 0].tolist()+ edges[:, 1].tolist()
    tally = defaultdict(list)
    for i, item in enumerate(l):
        tally[item].append(i if i < len(l)//2 else i - len(l)//2)
    
    output = []
    for _, locs in tally.items():
        if len(locs) > 1:
            output.append(list(combinations(locs, 2)))
    
    return torch.LongTensor(list(chain(*output))).to(edges.device)