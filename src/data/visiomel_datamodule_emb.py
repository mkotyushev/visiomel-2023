import logging
import pickle
from typing import List, Optional
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold

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
        self.data['file_index'] = self.data.groupby('path').ngroup()

    @property
    def targets(self):
        return self.data['label'].groupby('file_index').first().values
    
    @property
    def file_indices(self):
        return self.data['file_index'].values

    def __len__(self):
        return self.data['file_index'].nunique()

    def __getitem__(self, index):
        data = self.data[self.data['file_index'] == index]
        return data['features'].values, data['label'].values[0]


class VisiomelDatamodule(LightningDataModule):
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
    ):
        super().__init__()
        self.save_hyperparameters()

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert k is None or (0 <= fold_index < k), "incorrect fold number"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
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
                split = list(kfold.split(dataset, dataset.targets, dataset.file_indices))
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
            shuffle=shuffle
        )

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False
        )
        val_dataloader_downsampled = DataLoader(
            dataset=self.val_dataset_downsampled, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False
        )
        return [val_dataloader, val_dataloader_downsampled]