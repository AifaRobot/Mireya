import torch
import matplotlib.pyplot as plt
import numpy as np
import json

def draw_plot(history, xlabel, ylabel, save_path):
    new_array = [
        np.array(history[max(0, i - 100):i + 1]).mean()
        for i in range(len(history))
    ]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(new_array, "red")
    plt.title(ylabel)
    plt.savefig(save_path + '/' + ylabel + '.png')
    plt.close()

def open_jsonl_file(path_file):
    data = []
    
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def parse_jsonl_data_to_dict(tokens_jsonl):
    dict = {}

    for word in tokens_jsonl:
        dict[word['word']] = word['id']
    
    return dict

def generate_causal_mask(input):
    
    """Compares each element of input with 0. Example: If input = [[1, 2, 3, 0, 0]], then: [[True, True, True, False, False]]"""
    input_mask = (input != 0).unsqueeze(1).unsqueeze(3)

    sequence_length = input.size(1)
    
    """Creates a tensor filled with ones of form (1, sequence_length, sequence_length). This tensor will be the basis for
    building the causal mask.

    Extracts the top part of the triangular matrix, excluding the main diagonal. For sequence_length = 4, the result will be:

    [[0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]]

    The values are inverted (0 → 1, 1 → 0). This creates a lower triangular mask:

    [[1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]]

    Converts numeric values to boolean (1 → True, 0 → False):

    [[True, False, False, False],
    [True,  True, False, False],
    [True,  True,  True, False],
    [True,  True,  True,  True]]

    The purpose of the nopeal_mask is to ensure that a token at position i can only "see" previous tokens (or itself) and
    not future ones."""
    nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool().to(input_mask.device)
    
    """Combine the fill mask (output_mask) and the causal mask (nopeak_mask) using a logical "AND" operation.

    This ensures that padding (0) tokens are ignored and the causality constraint (don't look into the future) is respected.

    Example: If output_mask = [[[[True], [True], [True], [False]]]] and nopeak_mask is:

    [[True, False, False, False],
    [True,  True, False, False],
    [True,  True,  True, False],
    [True,  True,  True,  True]]

    The combined result (output_mask) will be:

    [[[[True, False, False, False],
    [True,  True, False, False],
    [True,  True,  True, False],
    [False, False, False, False]]]]"""
    input_mask = input_mask & nopeak_mask

    return input_mask