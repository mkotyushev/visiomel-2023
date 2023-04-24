import logging
import random
import numpy as np
import torch
from collections import Counter, defaultdict
from copy import deepcopy
from multiprocessing import Manager
from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from timm.data import rand_augment_transform
from torchvision.transforms import (
    Compose, 
    Resize, 
    ToTensor, 
    Normalize, 
    RandomCrop, 
    RandomHorizontalFlip
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data._utils.collate import default_collate

from src.data.transforms import Shrink, CenterCropPct, SimMIMTransform, PadCenterCrop
from src.utils.utils import loader_with_filepath


logger = logging.getLogger(__name__)


class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            x = self.transform(sample[0])
        return (x, *sample[1:])
        
    def __len__(self):
        return len(self.subset)


class VisiomelImageFolder(ImageFolder):
    def __init__(
        self, 
        root: str, 
        shared_cache=None, 
        pre_transform=None, 
        transform=None, 
        target_transform=None, 
        loader=None, 
        is_valid_file=None
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.cache = shared_cache
        self.pre_transform = pre_transform

    def load_cached(self, path):
        if self.cache is None or path not in self.cache:
            sample = self.loader(path)
            if self.pre_transform is not None:
                sample = self.pre_transform(sample)
            if self.cache is not None:
                self.cache[path] = sample
        else:
            sample = self.cache[path]
        return sample

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.load_cached(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def build_weighted_sampler(dataset):
    if isinstance(dataset, SubsetDataset):
        targets = [
            dataset.subset.dataset.targets[index] 
            for index in dataset.subset.indices
        ]
    elif isinstance(dataset, Subset):
        targets = [
            dataset.dataset.targets[index] 
            for index in dataset.indices
        ]
    elif isinstance(dataset, ImageFolder):
        targets = dataset.targets
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    target_counter = Counter(targets)
    class_weights = {k: 1 / v for k, v in target_counter.items()}
    weights = [class_weights[t] for t in targets]

    # num_samples is equal to the number of samples in the largest class
    # multiplied by the number of classes
    num_samples = max(target_counter.values()) * len(target_counter)

    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def build_downsampled_dataset(subset_dataset: SubsetDataset):
    # Here we need to copy the dataset to avoid changing 
    # the original one while preserving underlying VisiomelImageFolder
    # dataset to keep shared cache.
    cached_dataset = subset_dataset.subset.dataset
    subset_dataset = deepcopy(subset_dataset)
    subset_dataset.subset.dataset = cached_dataset

    # Get target to indices list mapping
    target_to_indices = defaultdict(list)
    for index in subset_dataset.subset.indices:
        target = subset_dataset.subset.dataset.targets[index]
        target_to_indices[target].append(index)

    # Downsample each class to the minimum number of samples
    min_target_num_samples = min(map(len, target_to_indices.values()))
    for target in target_to_indices:
        if len(target_to_indices[target]) > min_target_num_samples:
            target_to_indices[target] = random.sample(target_to_indices[target], min_target_num_samples)
    
    # Update indices
    subset_dataset.subset.indices = []
    for indices in target_to_indices.values():
        subset_dataset.subset.indices.extend(indices)
    
    return subset_dataset


class IdentityTransform:
    def __call__(self, x):
        return x


def simmim_collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class VisiomelDatamodule(LightningDataModule):
    def __init__(
        self,
        task: str = 'classification',
        data_dir_train: str = './data/train',	
        k: int = None,
        fold_index: int = 0,
        data_dir_test: Optional[str] = None,
        img_size: int = 224,
        shrink_preview_scale: Optional[int] = None,
        batch_size: int = 32,
        split_seed: int = 0,
        num_workers: int = 0,
        num_workers_saturated: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        sampler: Optional[str] = None,
        enable_caching: bool = False,
        data_shrinked: bool = False,
        train_resize_type: str = 'resize',
        mask_patch_size: int = 32,
        model_patch_size: int = 4,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert k is None or (0 <= fold_index < k), "incorrect fold number"
        
        # Caching
        self.shared_cache = None
        if enable_caching:
            manager = Manager()
            self.shared_cache = manager.dict()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.collate_fn = simmim_collate_fn if task == 'simmim' else default_collate
    
    def build_transforms(self):
        """Build data transformations."""
        img_mean = (238, 231, 234)  # from all train data
        if self.hparams.train_resize_type == 'resize':
            # - resize to img_size in pretransform
            # - do nothing on train
            # - do nothing on val
            pre_resize_transform = Resize(size=(self.hparams.img_size, self.hparams.img_size))
            train_resize_transform = val_resize_transform = IdentityTransform()
        elif self.hparams.train_resize_type == 'random_crop':
            # - do nothing in pretransform
            # - random crop to img_size on train
            # - center crop to img_size on val
            
            # Note: here crops could lead to empty areas, 
            # so padding is needed.
            
            # Note: image size after pretransform is rather large,
            # so caching probably is not possible.
            
            # Note: reflective padding is dataleak for SimMIM
            # and could not be used here
            # but not for classification.
            if self.hparams.enable_caching:
                logger.warning(
                    'Caching is enabled with large images. '
                    'Consider using "resize" train_resize_type.'
                )
            pre_resize_transform = IdentityTransform()
            train_resize_transform = RandomCrop(
                size=(self.hparams.img_size, self.hparams.img_size),
                pad_if_needed=True,
                padding_mode='constant',
                fill=img_mean
            )
            val_resize_transform = PadCenterCrop(
                size=(self.hparams.img_size, self.hparams.img_size),
                pad_if_needed=True,
                padding_mode='constant',
                fill=img_mean
            )
        elif self.hparams.train_resize_type == 'none':
            pre_resize_transform, train_resize_transform, val_resize_transform = \
                IdentityTransform(), IdentityTransform(), IdentityTransform()

        if self.hparams.data_shrinked:
            self.pre_transform = pre_resize_transform
        else:
            self.pre_transform = Compose(
                [
                    CenterCropPct(size=(0.9, 0.9)),
                    Shrink(scale=self.hparams.shrink_preview_scale, fill=img_mean),
                    pre_resize_transform,
                ]
            )

        if self.hparams.task == 'simmim':
            train_random_transform = RandomHorizontalFlip()
            mask_generator = MaskGenerator(
                input_size=self.hparams.img_size,
                mask_patch_size=self.hparams.mask_patch_size,
                model_patch_size=self.hparams.model_patch_size,
                mask_ratio=self.hparams.mask_ratio,
            )
            simmim_transform = SimMIMTransform(mask_generator)
        elif self.hparams.task == 'classification':
            train_random_transform = rand_augment_transform(
                config_str='rand-m9-mstd0.5',
                hparams=dict(img_mean=img_mean)
            )
            simmim_transform = IdentityTransform()
        elif self.hparams.task == 'raw':
            train_random_transform = IdentityTransform()
            simmim_transform = IdentityTransform()

        self.train_transform = Compose(
            [
                train_resize_transform,
                train_random_transform,
                ToTensor(),
                Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
                simmim_transform,
            ]
        )
        self.val_transform = self.test_transform = Compose(
            [
                val_resize_transform,
                ToTensor(),
                Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
                simmim_transform,
            ]
        )

    def setup(self, stage=None) -> None:
        """Setup data."""
        self.build_transforms()
        if self.train_dataset is None:
            # Train dataset
            if self.hparams.k is not None:
                dataset = VisiomelImageFolder(
                    self.hparams.data_dir_train, 
                    shared_cache=self.shared_cache,
                    pre_transform=self.pre_transform,
                    transform=None, 
                    loader=loader_with_filepath
                )
                kfold = StratifiedKFold(
                    n_splits=self.hparams.k, 
                    shuffle=True, 
                    random_state=self.hparams.split_seed
                )
                split = list(kfold.split(dataset, dataset.targets))
                train_indices, val_indices = split[self.hparams.fold_index]

                train_subset, val_subset = \
                    Subset(dataset, train_indices), Subset(dataset, val_indices)
                
                self.train_dataset, self.val_dataset = \
                    SubsetDataset(train_subset, transform=self.train_transform), \
                    SubsetDataset(val_subset, transform=self.val_transform)
                self.val_dataset_downsampled = build_downsampled_dataset(self.val_dataset)
            else:
                self.train_dataset = VisiomelImageFolder(
                    self.hparams.data_dir_train, 
                    shared_cache=self.shared_cache,
                    pre_transform=self.pre_transform,
                    transform=self.train_transform, 
                    loader=loader_with_filepath
                )
                self.val_dataset = None

            # Test dataset
            if self.hparams.data_dir_test is not None:
                self.test_dataset = VisiomelImageFolder(
                    self.hparams.data_dir_test, 
                    shared_cache=self.shared_cache,
                    pre_transform=self.pre_transform,
                    transform=self.test_transform, 
                    loader=loader_with_filepath
                )

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
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
            shuffle=False
        )
        val_dataloader_downsampled = DataLoader(
            dataset=self.val_dataset_downsampled, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        return [val_dataloader, val_dataloader_downsampled]

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "test dataset is not defined"
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
