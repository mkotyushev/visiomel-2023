import logging
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data._utils.collate import default_collate

from .visiomel_datamodule import SubsetDataset, build_downsampled_dataset, build_weighted_sampler
from src.utils.utils import load_embeddings


logger = logging.getLogger(__name__)


class EmbeddingDataset:
    def __init__(self, pkl_pathes: List[str]) -> None:
        self.data = pd.concat(load_embeddings(pkl_pathes).values())
        self.data['features'] = self.data['features'].apply(lambda x: np.array(x).squeeze(0))
        self.data['label'] = self.data['label'].apply(np.array)
        self.data['group'] = self.data['path']

    @property
    def targets(self):
        return self.data['label'].values
    
    @property
    def groups(self):
        return self.data['group'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row['features'], row['label'], row['path']


def split_fold(
    dataset_no_repeats: EmbeddingDataset, 
    dataset_with_repeats: EmbeddingDataset,
    k: int = 5,
    fold_index: int = 0,
    random_state: int = 0,
) -> Tuple[EmbeddingDataset, EmbeddingDataset, EmbeddingDataset, EmbeddingDataset]:
    kfold = StratifiedGroupKFold(
        n_splits=k, 
        shuffle=True, 
        random_state=random_state
    )
    split = list(kfold.split(dataset_no_repeats, dataset_no_repeats.targets.astype(int), dataset_no_repeats.groups))
    train_indices_no_repeats, test_indices_no_repeats = split[fold_index]

    # Get indices for embeddings with repeats by filenames (used for train)
    train_filenames = dataset_no_repeats.data.iloc[train_indices_no_repeats]['path'].values
    no_repeats_to_with_repeats_train_mask = dataset_with_repeats.data['path'].isin(train_filenames).values
             
    train_indices_with_repeats = np.arange(len(dataset_with_repeats))[
        no_repeats_to_with_repeats_train_mask
    ]
    test_indices_with_repeats = np.arange(len(dataset_with_repeats))[
        ~no_repeats_to_with_repeats_train_mask
    ]

    dataset_no_repeats_train = deepcopy(dataset_no_repeats)
    dataset_no_repeats_train.data = dataset_no_repeats_train.data.iloc[train_indices_no_repeats]
    dataset_no_repeats_test = deepcopy(dataset_no_repeats)
    dataset_no_repeats_test.data = dataset_no_repeats_test.data.iloc[test_indices_no_repeats]

    dataset_with_repeats_test = deepcopy(dataset_with_repeats)
    dataset_with_repeats_test.data = dataset_with_repeats_test.data.iloc[test_indices_with_repeats]
    dataset_with_repeats_train = deepcopy(dataset_with_repeats)
    dataset_with_repeats_train.data = dataset_with_repeats_train.data.iloc[train_indices_with_repeats]

    return \
        dataset_no_repeats_train, \
        dataset_no_repeats_test, \
        dataset_with_repeats_train, \
        dataset_with_repeats_test


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


def check_no_split_intersection(
    train_dataset: EmbeddingDataset, 
    val_dataset: EmbeddingDataset, 
    test_dataset: Optional[EmbeddingDataset] = None
):
    train_filenames = set(train_dataset.data['path'].values)
    val_filenames = set(val_dataset.data['path'].values)
    assert len(train_filenames & val_filenames) == 0, "train and val datasets intersect"

    if test_dataset is not None:
        test_filenames = set(test_dataset.data['path'].values)
        assert len(train_filenames & test_filenames) == 0, "train and test datasets intersect"
        assert len(val_filenames & test_filenames) == 0, "val and test datasets intersect"


class VisiomelDatamoduleEmb(LightningDataModule):
    def __init__(
        self,
        embedding_pathes: List[str],	
        embedding_pathes_aug_with_repeats: List[str],	
        batch_size: int = 32,
        k: int = None,
        fold_index: int = 0,
        k_test: Optional[int] = None,
        fold_index_test: int = 0,
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
            if self.hparams.k is not None:
                dataset_no_repeats = EmbeddingDataset(self.hparams.embedding_pathes)
                dataset_with_repeats = EmbeddingDataset(self.hparams.embedding_pathes_aug_with_repeats)

                # Test and val should be without repeats
                # train should be with repeats

                # split_fold returns:
                #     dataset_no_repeats_train, \
                #     dataset_no_repeats_test, \
                #     dataset_with_repeats_train, \
                #     dataset_with_repeats_test

                if self.hparams.k_test is not None:
                    (
                        dataset_no_repeats, 
                        self.test_dataset, 
                        dataset_with_repeats, 
                        _
                    ) = split_fold(
                        dataset_no_repeats, 
                        dataset_with_repeats, 
                        self.hparams.k_test, 
                        self.hparams.fold_index_test, 
                        self.hparams.split_seed
                    )
                
                (
                    _, 
                    self.val_dataset, 
                    self.train_dataset, 
                    _
                ) = split_fold(
                    dataset_no_repeats, 
                    dataset_with_repeats, 
                    self.hparams.k_test, 
                    self.hparams.fold_index_test, 
                    self.hparams.split_seed
                )

                # Check that train and val datasets do not intersect
                # in terms of filenames
                check_no_split_intersection(self.train_dataset, self.val_dataset, self.test_dataset)
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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
