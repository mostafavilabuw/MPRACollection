"""Main module."""

import yaml
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import data_to_XYobs, XYobs_to_data, mkdir, seqs_to_onehot

__SEED__ = 42
__PATH__ = "/data/tuxm/project/MPRA-collection/data/mpra_test/"



class MPRA_Dataset:

    @staticmethod
    def list_papers(folder=__PATH__):
        return os.listdir(folder)

    @staticmethod
    def list_datasets(name_paper, folder=__PATH__):
        return [name[:-4] for name in os.listdir(os.path.join(folder, name_paper)) if '.csv' in name]

    @staticmethod
    def print_all(folder=__PATH__):
        for name_paper in MPRA_Dataset.list_papers(folder):
            print(f'==== {name_paper} ====')
            for name_dataset in MPRA_Dataset.list_datasets(name_paper, folder):
                print(f'  ==== {name_dataset}')
        print(f'==== ==== ==== ====')

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
        Z=dict(),
        names_Z=list(),
        _mpra_dataset=None,
    ):
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
        
        if Z and names_Z:
            raise ValueError(
                "ONLY ONE of 'Z' or 'names_Z' should be provided, but not BOTH."
            )
        elif Z:
            for name in Z.keys():
                assert Z[name].shape[0] == self.n_seq
            self.Z = Z
        elif names_Z:
            self.Z = self.load_Z(names_Z)
        else:
            self.Z = dict()
        
        # TODO: inherit from _mpra_dataset (type: MPRA_Dataset) when provided
        self.folder = folder
        self.name_paper = name_paper
        self.name_dataset = name_dataset

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_seq(self):
        return self.X.shape[0]

    @property
    def len_max(self):
        return self.X['X'].str.len().max()
    
    @property
    def dim_seq(self):
        return 4

    @property
    def n_readout(self):
        return self.Y.shape[1]
    
    @property
    def names_readout(self):
        return [col for col in self.Y.columns]
    
    @property
    def n_embed(self):
        return len(self.Z)
    
    @property
    def names_embed(self):
        return [name for name in self.Z.keys()]

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
        embed_names = [name for name in self.Z.keys()]

        # Displaying observable and readout columns
        description += "    obs seq: '" + "', '".join(obs_seq_columns) + "'\n"
        description += "    obs readout: '" + "', '".join(obs_readout_columns) + "'\n"
        description += "    readout: '" + "', '".join(readout_columns) + "'\n"
        description += "    embed: '" + "', '".join(embed_names) + "'\n"
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
    
    def load_Z(self, names_Z: list):
        """Loads embeddings from files and stores them in the Z attribute."""

        if isinstance(names_Z, str):
            names_Z = [names_Z]
        for name in names_Z:
            try:
                self.Z[name] = torch.load(os.path.join(self.folder, self.name_paper, self.name_dataset, name + '.pt'))
                assert self.Z[name].shape[0] == self.n_seq
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Unable to load embeddings: {e}")
            except Exception as e:
                raise Exception(f"An error occurred while loading the embeddings: {e}")
    
    def save_Z(self, names_Z: list):
        """Saves embeddings to files based on the names provided."""
        mkdir(os.path.join(self.folder, self.name_paper, self.name_dataset))
        if isinstance(names_Z, str):
            names_Z = [names_Z]
        for name in names_Z:
            try:
                torch.save(self.Z[name], os.path.join(self.folder, self.name_paper, self.name_dataset, name + '.pt'))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Unable to save embeddings: {e}")
            except Exception as e:
                raise Exception(f"An error occurred while saving the embeddings: {e}")

    # PyTorch-related
    def to_Dataset(self, cols_Y: list = [], names_Z: list = [], len_max: int = None):
        cols_Y = (
            cols_Y
            if cols_Y
            else [col for col in self.data.columns if col.startswith("Y: ")]
        )

        # FIXME: change the hard-coded "3:" to a more general way
        cols_Y = [col[3:] if col.startswith("Y: ") else col for col in cols_Y]

        # TODO: should not directly delete the rows with missing values without warning
        mask = self.Y[cols_Y].notna().all(axis=1)
        len_max = len_max if len_max else self.len_max
        _X = torch.Tensor(
            seqs_to_onehot(self.X["X"][mask].values, len_max=len_max)
        ).transpose(1, 2)
        _Y = torch.Tensor(self.Y[cols_Y][mask].values)

        _Z = torch.zeros((_X.shape[0], 0))
        for name in names_Z:
            _Z = torch.cat((_Z, self.Z[name][mask]), dim=1)

        return TensorDataset(_X, _Y, _Z)

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

    # TODO: handle setitem
    # TODO: handle str or list of str for index (only apply index on Y, not X or Z)
    def __getitem__(self, index):

        if isinstance(index, str) or (isinstance(index, list) and all(isinstance(_index, str) for _index in index)):
            if isinstance(index, str):
                index = [index]
            try:
                _Y = self.Y[index].copy()
            except KeyError:
                raise KeyError("Provided index is out of bounds or invalid.")
            _X = self.X.copy()
            _obs_X = self.obs_X.copy()
            _obs_Y = self.obs_Y.copy()
            _Z = self.Z.copy()
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, self.info, 
                X=_X, Y=_Y, obs_X=_obs_X, obs_Y=_obs_Y, Z=_Z, 
            )

        elif isinstance(index, (int, slice, pd.Series, pd.Index, np.ndarray, torch.Tensor)) \
            or (isinstance(index, list) and all(isinstance(_index, int) for _index in index)):
            try:
                if isinstance(index, int):
                    _data = self.data.iloc[index].copy()
                elif isinstance(index, slice):
                    _data = self.data.iloc[index].copy()
                elif isinstance(index, list):
                    _data = self.data.iloc[index].copy()
                
                elif isinstance(index, pd.Series):
                    _data = self.data.loc[index].copy()
                elif isinstance(index, pd.Index):
                    _data = self.data.loc[index].copy()
                    
                elif isinstance(index, torch.Tensor):
                    index = index.numpy()
                    if index.dtype == bool:
                        _data = self.data.loc[index].copy()
                    elif index.dtype == int:
                        _data = self.data.iloc[index].copy()
                elif isinstance(index, np.ndarray):
                    if index.dtype == bool:
                        _data = self.data.loc[index].copy()
                    elif index.dtype == int:
                        _data = self.data.iloc[index].copy()

                else:
                    raise TypeError(f"Unsupported index type: {type(index).__name__}")
                
            except KeyError:
                raise KeyError("Provided index is out of bounds or invalid.")
            _data = _data.reset_index(drop=True)

            _Z = dict()
            for name in self.Z.keys():
                if isinstance(index, pd.Series) or isinstance(index, pd.Index):
                    index = index.to_numpy()
                _Z[name] = self.Z[name][index]
                
            # Create a new MPRA_Dataset with the selected data
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, self.info, data=_data, Z=_Z
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

    # @property
    # def embed(self):
    #     return self.Z
    
    # @embed.setter
    # def embed(self, value):
    #     self.Z = value
