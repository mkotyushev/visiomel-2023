from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from timm.data import rand_augment_transform
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.data.transforms import Shrink, CenterCropPct


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
        pin_memory: bool = False
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= fold_index < k, "incorrect fold number"
        
        # data transformations
        self.train_transform = Compose(
            [
                CenterCropPct(size=(0.9, 0.9)),
                Shrink(scale=shrink_preview_scale),
                Resize(size=(img_size, img_size)),
                rand_augment_transform(
                    config_str='rand-m9-mstd0.5',
                    hparams=dict(img_mean=(238, 231, 234))  # from train data
                ),
                ToTensor(),
                Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
            ]
        )

        non_train_transform = Compose(
            [
                CenterCropPct(size=(0.9, 0.9)),
                Shrink(scale=shrink_preview_scale),
                Resize(size=(img_size, img_size)),
                ToTensor(),
                Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
            ]
        )
        self.val_transform = non_train_transform
        self.test_transform = non_train_transform

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage=None) -> None:
        if self.train_dataset is None and self.val_dataset is None:
            # Train & val dataset as k-th fold
            dataset = ImageFolder(
                self.hparams.data_dir_train, transform=None)

            kfold = KFold(n_splits=self.hparams.k, shuffle=True, random_state=self.hparams.split_seed)
            split = list(kfold.split(dataset))
            train_indices, val_indices = split[self.hparams.fold_index]

            train_subset, val_subset = \
                Subset(dataset, train_indices), Subset(dataset, val_indices)
            
            self.train_dataset, self.val_dataset = \
                SubsetDataset(train_subset, transform=self.train_transform), \
                SubsetDataset(val_subset, transform=self.val_transform)

        # Test dataset
        if self.test_dataset is None and self.hparams.data_dir_test is not None:
            self.test_dataset = ImageFolder(
                self.hparams.data_dir_test, transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=2,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=2,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "test dataset is not defined"
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=2,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
