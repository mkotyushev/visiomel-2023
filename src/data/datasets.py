import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.utils.utils import load_embeddings


class SubsetDataset(Dataset):
    def __init__(
        self, 
        subset, 
        transform=None,
        n_repeats=1,
    ):
        self.subset = subset
        self.transform = transform
        self.n_repeats = n_repeats
        
    def __getitem__(self, index):
        sample = self.subset[index]

        samples = []
        for _ in range(self.n_repeats):
            for sample in self.subset[index]:
                if self.transform:
                    x = (self.transform(sample[0]), *sample[1:])
                else:
                    x = sample
                samples.append(x)
            
        return samples
        
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
        is_valid_file=None,
        n_repeats=1,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.cache = shared_cache
        self.pre_transform = pre_transform
        self.n_repeats = n_repeats

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

        samples = []
        for _ in range(self.n_repeats):
            if self.transform is not None:
                x = self.transform(sample)
            else:
                x = sample
            if self.target_transform is not None:
                y = self.target_transform(target)
            else:
                y = target

            samples.append((x, y, path))

        return samples



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
