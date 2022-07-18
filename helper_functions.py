import numpy as np

# ------------------------- ordinal encoding ----------------------------------------------------------------------
protein_dict = {"Q": 1, "N": 2, "K": 3, "W": 4, "F": 5, "P": 6, "Y": 7, "L": 8, "M": 9, "T": 10, "E": 11, "I": 12,
                "A": 13, "R": 14, "G": 15, "H": 16, "S": 17, "D": 18, "V": 19, "C": 20}

reverse_dict = {1: "Q", 2: "N", 3: "K", 4: "W", 5: "F", 6: "P", 7: "Y", 8: "L", 9: "M", 10: "T", 11: "E", 12: "I",
                13: "A", 14: "R", 15: "G", 16: "H", 17: "S", 18: "D", 19: "V", 20: "C"}


def ordinal_enc(protein_seq: str):
    """encode proteins as numbers 0-20 """
    res = []
    for i in protein_seq:
        res.append(protein_dict[i])
    return np.asarray(res)


def encode_list(seq):
    encoded_seq = []
    for i in range(0, len(seq)):
        encoded_seq.append(ordinal_enc(seq[i]))
    return np.asarray(encoded_seq)


def kmer_encoder(protein_seq: str, k: int):
    """"take a string encoding a protein sequence and an integer k and return all substrings (-sequences) of length k"""
    n = len(protein_seq)
    res = []

    for i in range(0, n - k + 1):
        res.append(protein_seq[i:i+k])

    return res


# ----------------------- evaluation functions ------------------------------------------------------------------

def confusion_matrix(pred_labels, actual_labels, num_clades = 17):
    # initialize empty 18 x 18 matrix
    matrix = [[0 for col in range(num_clades + 1)] for row in range(num_clades + 1)]
    # y-axis: actual clades, x-axis: predicted clades
    for i in range(0, len(pred_labels)):
        matrix[pred_labels[i]][actual_labels[i]] += 1
    return matrix


def accuracy(pred_labels, actual_labels):
    total = len(actual_labels)
    correct = 0
    for i in range(0,len(actual_labels)):
        if pred_labels[i] == actual_labels[i]:
            correct += 1
    return correct/total
