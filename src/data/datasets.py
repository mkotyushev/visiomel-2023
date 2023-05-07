import numpy as np
import pandas as pd
from typing import List, Optional
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


def preprocess_meta(df, debug=False):
    # keep age	sex	body_site	melanoma_history cols
    df = df.drop(columns=df.columns.difference(['age', 'sex', 'body_site', 'melanoma_history']))

    # Age: remove nans, convert to float, cut and encode interval by int
    df['age'] = df['age'].fillna('[0:0[')
    df['age'] = df['age'].apply(lambda x: x.split(':')[0][1:]).astype(float)
    df['age'] = pd.cut(df['age'], bins=[-10.0, 50.0, 63.0, 72.0, 1000.0], labels=False)

    if debug:
        assert df['age'].isna().sum() == 0
        assert (df['age'].min() == 0) & (df['age'].max() == 3)
        assert df['age'].dtype == int

    # Sex: remove nans, encode as int
    df['sex'] = df['sex'].fillna(1)
    df['sex'] = df['sex'] - 1
    if debug:
        assert df['sex'].isna().sum() == 0
        assert (df['sex'].min() == 0) & (df['sex'].max() == 1)
        assert df['sex'].dtype == int

    # Body site: remove nans, encode as int
    df['body_site'] = df['body_site'].fillna('unknown')
    mask_0 = (
        (df.body_site == 'upper limb/shoulder') | 
        (df.body_site == 'arm') | 
        (df.body_site == 'forearm') | 
        (df.body_site == 'hand') | 
        (df.body_site == 'hand/foot/nail') | 
        (df.body_site == 'nail') | (df.body_site == 'finger')
    )
    mask_1 = ((df.body_site == 'trunc') | (df.body_site == 'trunk'))
    mask_2 = (
        (df.body_site == 'head/neck') | 
        (df.body_site == 'face') | 
        (df.body_site == 'neck') | 
        (df.body_site == 'scalp')
    )
    mask_3 = (
        (df.body_site == 'leg') | 
        (df.body_site == 'lower limb/hip') | 
        (df.body_site == 'thigh') | 
        (df.body_site == 'foot') | 
        (df.body_site == 'toe') | 
        (df.body_site == 'sole') | 
        (df.body_site == 'seat')
    )
    df['body_site_int'] = 0  # unknown are also 0
    df.loc[mask_0, 'body_site_int'] = 0
    df.loc[mask_1, 'body_site_int'] = 1
    df.loc[mask_2, 'body_site_int'] = 2
    df.loc[mask_3, 'body_site_int'] = 3
    df['body_site'] = df['body_site_int']
    df = df.drop(columns=['body_site_int'])
    if debug:
        assert df['body_site'].isna().sum() == 0
        assert (df['body_site'].min() == 0) & (df['body_site'].max() == 3)
        assert df['body_site'].dtype == int

    # Melanoma history: remove nans, encode as int
    df['melanoma_history'] = df['melanoma_history'].fillna('UNK')
    mask_0 = (df['melanoma_history'] == 'NO')
    mask_1 = (df['melanoma_history'] == 'YES')
    mask_2 = (df['melanoma_history'] == 'UNK')
    df['melanoma_history_int'] = 2  # unknown are 2
    df.loc[mask_0, 'melanoma_history_int'] = 0
    df.loc[mask_1, 'melanoma_history_int'] = 1
    df['melanoma_history'] = df['melanoma_history_int']
    df = df.drop(columns=['melanoma_history_int'])
    if debug:
        assert df['melanoma_history'].isna().sum() == 0
        assert (df['melanoma_history'].min() == 0) & (df['melanoma_history'].max() == 2)
        assert df['melanoma_history'].dtype == int
    
    return df


class EmbeddingDataset:
    def __init__(self, pkl_pathes: List[str], meta_filepath: Optional[str]) -> None:
        self.data = pd.concat(load_embeddings(pkl_pathes).values())
        self.data['features'] = self.data['features'].apply(lambda x: np.array(x).squeeze(0))
        self.data['label'] = self.data['label'].apply(np.array)
        self.data['group'] = self.data['path']
        self.meta = None
        if meta_filepath is not None:
            self.meta = preprocess_meta(pd.read_csv(meta_filepath))
            # extract filename replace .png with .tif if needed
            # to match with the meta data
            self.data['filename'] = self.data['path'].apply(lambda x: x.name.replace('.png', '.tif'))
            self.data = self.data.merge(self.meta, on='filename', how='left')
            self.data = self.data.drop(columns=['filename'])

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
        if self.meta is None:
            # x, y, cache_key
            return row['features'], row['label'], row['path']
        # x, meta, y, cache_key
        # meta[:, 0] - age
        # meta[:, 1] - sex
        # meta[:, 2] - body_site
        # meta[:, 3] - melanoma_history
        return \
            row['features'], \
            row[['age', 'sex', 'body_site', 'melanoma_history']], \
            row['label'], \
            row['path']
