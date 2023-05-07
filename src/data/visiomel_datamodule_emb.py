import logging
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold, StratifiedGroupKFold
from torch.utils.data._utils.collate import default_collate

from .visiomel_datamodule import build_downsampled_dataset, build_weighted_sampler
from .datasets import EmbeddingDataset, SubsetDataset

logger = logging.getLogger(__name__)


def build_downsampled_datasets(
    dataset: Union[SubsetDataset, Dataset], 
    k: int = 5, 
    random_state: int = 0,
    method: str = 'kfold',
) -> List[SubsetDataset]:
    assert method in ['kfold', 'bootstrap'], "incorrect method"

    # Wrap dataset into SubsetDataset if needed
    if isinstance(dataset, SubsetDataset):
        subset_dataset = dataset
    else:
        subset = Subset(dataset, indices=range(len(dataset)))
        subset_dataset = SubsetDataset(subset, transform=None, n_repeats=1)

    # Get target to indices list mapping
    target_to_indices = defaultdict(list)
    for index in subset_dataset.subset.indices:
        target = subset_dataset.subset.dataset.targets[index]
        target_to_indices[target].append(index)

    assert len(target_to_indices) == 2, "Dataset must have at 2 classes"
    assert len(target_to_indices[0]) >= len(target_to_indices[1]), \
        "Dataset must have more negative samples than positive ones"
    
    # Preserving underlying VisiomelImageFolder
    # dataset to keep shared cache.
    cached_dataset = subset_dataset.subset.dataset

    subset_datasets = []

    if method == 'kfold':
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for _, keep_negative_indices in kfold.split(target_to_indices[0]):
            # Here we need to copy the dataset to avoid changing 
            # the original one
            subset_dataset_current = deepcopy(subset_dataset)
            subset_dataset_current.subset.dataset = cached_dataset

            # Update indices
            subset_dataset_current.subset.indices = [
                *[target_to_indices[0][i] for i in keep_negative_indices],
                *target_to_indices[1]
            ]
            subset_datasets.append(subset_dataset_current)
    else:
        # Bootstrap
        for _ in range(k):
            # Here we need to copy the dataset to avoid changing 
            # the original one
            subset_dataset_current = deepcopy(subset_dataset)
            subset_dataset_current.subset.dataset = cached_dataset

            # Update indices
            subset_dataset_current.subset.indices = [
                *np.random.choice(target_to_indices[0], size=len(target_to_indices[1]), replace=True),
                *target_to_indices[1]
            ]
            subset_datasets.append(subset_dataset_current)
    
    return subset_datasets


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

    X = [torch.from_numpy(x) for x, _, _, _ in batch]
    meta = default_collate([torch.from_numpy(meta_).long() for _, meta_, _, _ in batch])
    y = default_collate([y_ for _, _, y_, _ in batch])
    paths = [path for _, _, _, path in batch]

    lengths = torch.tensor([len(x) for x in X])
    mask = ~torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in lengths], batch_first=True).bool()
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

    # x, mask, meta, y, cache_key = batch
    return X, mask, meta, y, paths


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
        val_dataset_downsampled_k: int = 30,
        meta_filepath: Optional[str] = None,
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
                dataset_no_repeats = EmbeddingDataset(
                    self.hparams.embedding_pathes,
                    meta_filepath=self.hparams.meta_filepath
                )
                dataset_with_repeats = EmbeddingDataset(
                    self.hparams.embedding_pathes_aug_with_repeats,
                    meta_filepath=self.hparams.meta_filepath
                )

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
                    self.hparams.k, 
                    self.hparams.fold_index, 
                    self.hparams.split_seed
                )

                # Check that train and val datasets do not intersect
                # in terms of filenames
                check_no_split_intersection(self.train_dataset, self.val_dataset, self.test_dataset)
                self.val_dataset_downsampled = build_downsampled_datasets(
                    self.val_dataset, 
                    k=self.hparams.val_dataset_downsampled_k, 
                    random_state=self.hparams.split_seed,
                    method='bootstrap',
                )
            else:
                self.train_dataset = EmbeddingDataset(
                    self.hparams.embedding_pathes, 
                    meta_filepath=self.hparams.meta_filepath
                )
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

        val_dataloaders_downsampled = []
        for dataset in self.val_dataset_downsampled:
            dataloader = DataLoader(
                dataset=dataset, 
                batch_size=self.hparams.batch_size, 
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            val_dataloaders_downsampled.append(dataloader)
        return [val_dataloader, *val_dataloaders_downsampled]

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

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
