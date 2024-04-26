"""Main module."""

import yaml
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import data_to_XYobs, XYobs_to_data, mkdir, seqs_to_onehot

__PATH__ = "/data/tuxm/project/MPRA-collection/data/mpra_test/"


class MPRA_Dataset:

    def __init__(
        self,
        folder=__PATH__,
        name_paper="",
        name_dataset="",
        info=dict(),
        data=pd.DataFrame(),
        X=pd.DataFrame(),
        Y=pd.DataFrame(),
        obs_X=pd.DataFrame(),
        obs_Y=pd.DataFrame(),
    ):
        self.folder = folder
        self.name_paper = name_paper
        self.name_dataset = name_dataset

        self.info = info
        if data.shape == (0, 0) and X.shape == (0, 0):
            raise ValueError(
                "ONLY ONE of 'data' or 'X' should be provided, but not NEITHER."
            )
        elif data.shape == (0, 0):
            self.data = XYobs_to_data(X, Y, obs_X, obs_Y)
            self.X, self.Y, self.obs_X, self.obs_Y = X, Y, obs_X, obs_Y
        elif X.shape == (0, 0):
            self.data = data
            self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(data)
        else:
            raise ValueError(
                "ONLY ONE of 'data' or 'X' should be provided, but not BOTH."
            )

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_seq(self):
        return self.X.shape[0]

    @property
    def n_readout(self):
        return self.Y.shape[1]

    @property
    def n_seqXreadout(self):
        return self.n_seq * self.n_readout

    @property
    def n_readoutXseq(self):
        return self.n_readout * self.n_seq

    def __str__(self) -> str:
        # Basic dataset information
        description = f"MPRA_Dataset object with n_seq × n_readout = {self.n_seq} × {self.n_readout}\n"

        # Identifying observable and readout columns, assuming a naming convention is used.
        obs_seq_columns = [col for col in self.obs_X.columns]
        obs_readout_columns = [col for col in self.obs_Y.columns]
        readout_columns = [col for col in self.Y.columns]

        # Displaying observable and readout columns
        description += "    obs X: '" + "', '".join(obs_seq_columns) + "'\n"
        description += "    obs Y: '" + "', '".join(obs_readout_columns) + "'\n"
        description += "    readout: '" + "', '".join(readout_columns) + "'\n"
        # Displaying additional information in info
        description += "Additional information:\n"
        for key, value in self.info.items():
            description += f"    {key}: {value}\n"
        return description

    def __repr__(self):
        return self.__str__()

    # IO-related
    @staticmethod
    def load(name_paper: str, name_dataset: str, folder=__PATH__):
        """Loads dataset information and data from YAML and CSV files."""
        try:
            with open(
                os.path.join(folder, name_paper, f"{name_dataset}.yaml"), "r"
            ) as file:
                info = yaml.safe_load(file)
            data = pd.read_csv(os.path.join(folder, name_paper, f"{name_dataset}.csv"))
            return MPRA_Dataset(folder, name_paper, name_dataset, info, data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to load dataset: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the dataset: {e}")

    def reload(self):
        """Reloads dataset from source files based on current configuration."""
        try:
            file_path = os.path.join(self.folder, self.name_paper, self.name_dataset)
            with open(f"{file_path}.yaml", "r") as f:
                self.info = yaml.safe_load(f)
            self.data = pd.read_csv(f"{file_path}.csv")
            self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(self.data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to reload dataset: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while reloading the dataset: {e}")

    def save(self):
        """Saves the current dataset state to YAML and CSV files."""

        mkdir(os.path.join(self.folder, self.name_paper))
        file = os.path.join(self.folder, self.name_paper, self.name_dataset)
        with open(file + ".yaml", "w") as f:
            yaml.safe_dump(self.info, f)
        self.data.to_csv(file + ".csv", index=False)

    # PyTorch-related
    def to_Dataset(self, cols_Y: list = []):
        cols_Y = (
            cols_Y
            if cols_Y
            else [col for col in self.data.columns if col.startswith("Y: ")]
        )

        # FIXME: change the hard-coded "3:" to a more general way
        cols_Y = [col[3:] if col.startswith("Y: ") else col for col in cols_Y]

        # TODO: should not directly delete the rows with missing values without warning
        mask = self.Y[cols_Y].notna().all(axis=1)
        len_max = self.X["X"][mask].str.len().max()
        _X = torch.Tensor(
            seqs_to_onehot(self.X["X"][mask].values, len_max=len_max)
        ).transpose(1, 2)
        _Y = torch.Tensor(self.Y[cols_Y][mask].values)
        return TensorDataset(_X, _Y)

    def to_DataLoader(
        self,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        return DataLoader(
            self.to_Dataset(*args, **kwargs),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def __getitem__(self, index):
        # Check if index is a simple type (int, str, slice) or a pandas compatible index (list, Series, array)
        if isinstance(
            index, (int, str, slice, list, pd.Series, pd.Index, np.ndarray, torch.Tensor)
        ):
            # Normalize torch.Tensor to list
            if isinstance(index, torch.Tensor):
                index = index.tolist()

            # Using pandas DataFrame indexing directly handles int, str, slice, list of ints, and boolean array
            try:
                _data = self.data.loc[index]
            except KeyError:
                raise KeyError("Provided index is out of bounds or invalid.")

            # Create a new MPRA_Dataset with the selected data
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, self.info, _data
            )
        else:
            raise TypeError(f"Unsupported index type: {type(index).__name__}")

    @property
    def seq(self):
        return self.X

    @seq.setter
    def seq(self, value):
        self.X = value

    @property
    def obs_seq(self):
        return self.obs_X

    @obs_seq.setter
    def obs_seq(self, value):
        self.obs_X = value

    @property
    def readout(self):
        return self.Y

    @readout.setter
    def readout(self, value):
        self.Y = value

    @property
    def obs_readout(self):
        return self.obs_Y

    @obs_readout.setter
    def obs_readout(self, value):
        self.obs_Y = value
