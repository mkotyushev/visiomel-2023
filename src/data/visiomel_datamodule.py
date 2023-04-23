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
    CenterCrop,
    RandomHorizontalFlip
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data._utils.collate import default_collate

from src.data.transforms import Shrink, CenterCropPct
from src.utils.utils import loader_with_filepath


logger = logging.getLogger(__name__)


class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
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

        return sample, target


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


class VisiomelTrainDatamodule(LightningDataModule):
    def __init__(
        self,
        data_dir_train: str = './data/train',	
        k: int = 5,
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
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= fold_index < k, "incorrect fold number"
        
        # Caching
        self.shared_cache = None
        if enable_caching:
            manager = Manager()
            self.shared_cache = manager.dict()
        
        self.collate_fn = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def build_transforms(self):
        """Build task-specific data transformations."""
    
    def setup(self, stage=None) -> None:
        """Setup data."""
        self.build_transforms()

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


class VisiomelTrainDatamoduleClassification(VisiomelTrainDatamodule):
    def __init__(
        self,
        data_dir_train: str = './data/train',	
        k: int = 5,
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
    ):
        super().__init__(
            data_dir_train=data_dir_train,
            k=k,
            fold_index=fold_index,
            data_dir_test=data_dir_test,
            img_size=img_size,
            shrink_preview_scale=shrink_preview_scale,
            batch_size=batch_size,
            split_seed=split_seed,
            num_workers=num_workers,
            num_workers_saturated=num_workers_saturated,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            sampler=sampler,
            enable_caching=enable_caching,
            data_shrinked=data_shrinked,
            train_resize_type=train_resize_type
        )
        self.save_hyperparameters()
    
    def build_transforms(self):
        """Build task-specific data transformations."""
        if self.hparams.train_resize_type == 'resize':
            # Resize could be used for caching, so use it in pre_transform
            resize_transform_train = resize_transform_val = IdentityTransform()
            resize_transform_pre_transform = Resize(size=(self.hparams.img_size, self.hparams.img_size))
        elif self.hparams.train_resize_type == 'random_crop':
            # RandomCrop is not suitable for caching, so use it in train_transform
            # and do not resize in pre_transform. If enable_caching is set, 
            # it will cache large images before resize.
            if self.hparams.enable_caching:
                logger.warning(
                    'Caching is enabled with large images. '
                    'Consider using "resize" train_resize_type.'
                )
            resize_transform_train = RandomCrop(
                size=(self.hparams.img_size, self.hparams.img_size),
                pad_if_needed=True,
                padding_mode='reflect'
            )
            resize_transform_val = CenterCrop(size=(self.hparams.img_size, self.hparams.img_size))
            resize_transform_pre_transform = IdentityTransform()

        if self.hparams.data_shrinked:
            self.pre_transform = resize_transform_pre_transform
            img_mean = (193, 187, 205)  # from train data shrinked
        else:
            self.pre_transform = Compose(
                [
                    CenterCropPct(size=(0.9, 0.9)),
                    Shrink(scale=self.hparams.shrink_preview_scale),
                    resize_transform_pre_transform,
                ]
            )
            img_mean = (238, 231, 234)  # from all train data

        self.train_transform = Compose(
            [
                resize_transform_train,
                rand_augment_transform(
                    config_str='rand-m9-mstd0.5',
                    hparams=dict(img_mean=img_mean)
                ),
                ToTensor(),
                Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))
            ]
        )

        non_train_transform = Compose(
            [
                resize_transform_val,
                ToTensor(),
                Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD))
            ]
        )
        self.val_transform = non_train_transform
        self.test_transform = non_train_transform

    def setup(self, stage=None) -> None:
        """Setup data."""
        super().setup(stage)

        if self.train_dataset is None and self.val_dataset is None:
            # Train & val dataset as k-th fold
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

        # Test dataset
        if self.test_dataset is None and self.hparams.data_dir_test is not None:
            self.test_dataset = VisiomelImageFolder(
                self.hparams.data_dir_test, 
                shared_cache=None,
                pre_transform=self.pre_transform,
                transform=self.test_transform, 
                loader=loader_with_filepath
            )


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


class SimMIMTransform:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def __call__(self, img):
        mask = self.mask_generator()
        return img, mask


class VisiomelTrainDatamoduleSimMIM(VisiomelTrainDatamodule):
    def __init__(
        self,
        data_dir_train: str = './data/train',	
        k: int = 5,
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
        mask_ratio: float = 0.6
    ):
        super().__init__(
            data_dir_train=data_dir_train,
            k=k,
            fold_index=fold_index,
            data_dir_test=data_dir_test,
            img_size=img_size,
            shrink_preview_scale=shrink_preview_scale,
            batch_size=batch_size,
            split_seed=split_seed,
            num_workers=num_workers,
            num_workers_saturated=num_workers_saturated,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            sampler=sampler,
            enable_caching=enable_caching,
            data_shrinked=data_shrinked,
            train_resize_type=train_resize_type
        )
        self.save_hyperparameters()
        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )
        self.collate_fn = simmim_collate_fn
    
    def build_transforms(self):
        """Build task-specific data transformations."""
        assert self.hparams.train_resize_type == 'resize'
        resize_transform_pre_transform = Resize(size=(self.hparams.img_size, self.hparams.img_size))

        if self.hparams.data_shrinked:
            self.pre_transform = resize_transform_pre_transform
        else:
            self.pre_transform = Compose(
                [
                    CenterCropPct(size=(0.9, 0.9)),
                    Shrink(scale=self.hparams.shrink_preview_scale),
                    resize_transform_pre_transform,
                ]
            )

        self.transform = Compose(
            [
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
                SimMIMTransform(self.mask_generator),
            ]
        )

    def setup(self, stage=None) -> None:
        """Setup data."""
        super().setup(stage)
        if self.train_dataset is None:
            dataset = VisiomelImageFolder(
                self.hparams.data_dir_train, 
                shared_cache=self.shared_cache,
                pre_transform=self.pre_transform,
                transform=self.transform, 
                loader=loader_with_filepath
            )

            kfold = StratifiedKFold(
                n_splits=self.hparams.k, 
                shuffle=True, 
                random_state=self.hparams.split_seed
            )
            split = list(kfold.split(dataset, dataset.targets))
            train_indices, val_indices = split[self.hparams.fold_index]

            self.train_dataset, self.val_dataset = \
                Subset(dataset, train_indices), Subset(dataset, val_indices)


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
        return val_dataloader
    
    def test_dataloader(self) -> DataLoader:
        return None
