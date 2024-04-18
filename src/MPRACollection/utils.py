import os

import pandas as pd
import numpy as np


def data_to_XYobs(data):
    cols_X = ["X"] if "X" in data.columns else []
    cols_Y = [col for col in data.columns if col.startswith("Y: ")]
    cols_obs_X = [col for col in data.columns if col.startswith("obs_X: ")]
    cols_obs_Y = [col for col in data.columns if col.startswith("obs_Y: ")]
    X, Y, obs_X, obs_Y = data[cols_X], data[cols_Y], data[cols_obs_X], data[cols_obs_Y]

    # rename columns to be without prefix
    Y = Y.rename(columns=lambda col: col.replace("Y: ", ""))
    obs_X = obs_X.rename(columns=lambda col: col.replace("obs_X: ", ""))
    obs_Y = obs_Y.rename(columns=lambda col: col.replace("obs_Y: ", ""))
    return X, Y, obs_X, obs_Y


def XYobs_to_data(X, Y, obs_X, obs_Y):
    # add prefix to columns
    Y = Y.rename(columns=lambda col: "Y: " + col)
    obs_X = obs_X.rename(columns=lambda col: "obs_X: " + col)
    obs_Y = obs_Y.rename(columns=lambda col: "obs_Y: " + col)
    data = pd.concat([X, Y, obs_X, obs_Y], axis=1)
    return data


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def seqs_to_onehot(
    seqs,
    len_max=64,
    len_div=1,
):
    len_max = (len_max + len_div - 1) // len_div * len_div

    # Function to adjust sequence length to len_max
    def to_len_max(seq):
        if len(seq) < len_max:
            # Pad sequence if it's shorter than len_max
            padding = len_max - len(seq)
            return seq + "N" * padding
        else:
            # Truncate sequence if it's longer than len_max
            return seq[:len_max]

    seqs = [to_len_max(seq) for seq in seqs]

    alphabet = "ATGC"
    alphabet_unknown = "NX"
    value_unknown = 1.0 / len(alphabet)

    def to_uint8(seq):
        return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max + 1, len(alphabet)))
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet))
    hash_table[to_uint8(alphabet_unknown)] = value_unknown

    def seq_to_onehot(seq):
        return hash_table[to_uint8(seq)]

    seqs = [seq_to_onehot(seq) for seq in seqs]

    return np.stack(seqs)
