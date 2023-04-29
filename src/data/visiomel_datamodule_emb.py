import logging
import pickle
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data._utils.collate import default_collate

from .visiomel_datamodule import SubsetDataset, build_downsampled_dataset, build_weighted_sampler


logger = logging.getLogger(__name__)


class EmbeddingDataset:
    def __init__(self, pkl_pathes: List[str]) -> None:
        self.data = None
        for path in pkl_pathes:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if self.data is None:
                    self.data = data
                else:
                    self.data = pd.concat([self.data, data])
        self.data['features'] = self.data['features'].apply(lambda x: np.array(x).squeeze(0))
        self.data['label'] = self.data['label'].apply(np.array)
        self.data['file_index'] = self.data.groupby('path').ngroup()

    @property
    def targets(self):
        return self.data['label'].values
    
    @property
    def file_indices(self):
        return self.data['file_index'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row['features'], row['label'], row['path']
    

def masked_collate_fn(batch):
    # batch: list of [X: np.array, y: int, path: str] ->
    # X.shape = (n_frames, n_features), n_frames could 
    # be different for different samples

    # Pad & pack sequences with different lengths to max length
    # across the batch, create bool mask for padded values

    X = [torch.from_numpy(x) for x, _, _ in batch]
    y = default_collate([y_ for _, y_, _ in batch])
    paths = [path for _, _, path in batch]

    lengths = torch.tensor([len(x) for x in X])
    mask = ~torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in lengths], batch_first=True).bool()
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

    return X, mask, y, paths


class VisiomelDatamoduleEmb(LightningDataModule):
    def __init__(
        self,
        embedding_pathes: List[str],	
        batch_size: int = 32,
        k: int = None,
        fold_index: int = 0,
        split_seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        sampler: Optional[str] = None,
        num_workers_saturated: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert k is None or (0 <= fold_index < k), "incorrect fold number"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.collate_fn = masked_collate_fn
    
    def setup(self, stage=None) -> None:
        """Setup data."""
        if self.train_dataset is None:
            # Train dataset
            if self.hparams.k is not None:
                dataset = EmbeddingDataset(self.hparams.embedding_pathes)
                kfold = StratifiedGroupKFold(
                    n_splits=self.hparams.k, 
                    shuffle=True, 
                    random_state=self.hparams.split_seed
                )
                split = list(kfold.split(dataset, dataset.targets.astype(int), dataset.file_indices))
                train_indices, val_indices = split[self.hparams.fold_index]

                train_subset, val_subset = \
                    Subset(dataset, train_indices), Subset(dataset, val_indices)
                
                self.train_dataset, self.val_dataset = \
                    SubsetDataset(train_subset, transform=None, n_repeats=1), \
                    SubsetDataset(val_subset, transform=None, n_repeats=1)
                self.val_dataset_downsampled = build_downsampled_dataset(self.val_dataset)
            else:
                self.train_dataset = EmbeddingDataset(self.hparams.embedding_pathes)
                self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        sampler, shuffle = None, True
        if self.hparams.sampler is not None and self.hparams.sampler == 'weighted_upsampling':
            sampler = build_weighted_sampler(self.train_dataset)
            shuffle = False
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        val_dataloader_downsampled = DataLoader(
            dataset=self.val_dataset_downsampled, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        return [val_dataloader, val_dataloader_downsampled]
