# Remove PIL image size limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib
import numpy as np
import logging
import timm
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets.folder import default_loader
from typing import Dict, Optional, Union, Any, Literal
from model.patch_embed_with_backbone import SwinTransformerV2Modded
from torchmetrics import Metric
from torchmetrics.classification import BinaryFBetaScore
from torchmetrics.utilities.compute import _safe_divide
from mock import patch
from sklearn.metrics import log_loss
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from torch import Tensor
from weakref import proxy


logger = logging.getLogger(__name__)


# https://github.com/microsoft/Swin-Transformer/blob/f92123a0035930d89cf53fcb8257199481c4428d/utils.py
def load_pretrained(state_dict, model):
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    if 'head.bias' in state_dict:
        # check classifier, if not match, then re-init classifier to zero
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f'data/map22kto1k.txt'
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.)
                torch.nn.init.constant_(model.head.weight, 0.)
                del state_dict['head.weight']
                del state_dict['head.bias']
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    return model


def loader_with_filepath(path):
    """Load image and set filepath attribute."""
    img = default_loader(path)
    img.filepath = path
    return img


class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger):
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")


def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


def build_model(
    model_name, 
    num_classes, 
    img_size, 
    patch_size, 
    mock_class=SwinTransformerV2Modded,
    patch_embed_backbone_name=None,
    patch_embed_backbone_pretrained=True,
    pretrained=True,
    quadtree=False,
    drloc_params=None
):   
    # Load pretrained model with its default img_size
    # and then load the pretrained weights to the model via
    # load_pretrained function by Swin V2 authors
    pretrained_model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )
    patch_embed_backbone = None
    if patch_embed_backbone_name is not None:
        patch_embed_backbone = timm.create_model(
            patch_embed_backbone_name, 
            pretrained=patch_embed_backbone_pretrained, 
            num_classes=0
        )

    with patch('timm.models.swin_transformer_v2.SwinTransformerV2', mock_class):
        model = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=num_classes, 
            img_size=img_size, 
            patch_embed_backbone=patch_embed_backbone, 
            patch_size=patch_size,
            quadtree=quadtree,
            drloc_params=drloc_params,
        )
    model = load_pretrained(pretrained_model.state_dict(), model)

    del pretrained_model
    torch.cuda.empty_cache()

    return model


class LogLossScore(Metric):
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False
    def __init__(self):
        super().__init__()
        self.add_state("probas", default=[])
        self.add_state("target", default=[])

    def update(self, probas: torch.Tensor, target: torch.Tensor):
        assert probas.shape[0] == target.shape[0], \
            f'probas and target must have same first dimension, ' \
            f'got {probas.shape[0]} and {target.shape[0]}'

        self.probas.append(probas)
        self.target.append(target)

    def compute(self):
        self.probas = torch.cat(self.probas, dim=0)
        self.target = torch.cat(self.target, dim=0)
        if self.probas.isnan().any():
            return np.nan
        return log_loss(self.target.cpu().numpy(), self.probas.cpu().numpy(), eps=1e-16).item()


def extract_features_last_only_single(model, x):
    return model.forward_features(x)


def extract_features_single(model, x):
    x = model.patch_embed(x)
    if model.absolute_pos_embed is not None:
        x = x + model.absolute_pos_embed
    x = model.pos_drop(x)

    features = [x.mean(dim=1)]
    for layer in model.layers:
        x = layer(x)
        features.append(x.mean(dim=1))
    features[-1] = model.norm(features[-1])
    features = torch.cat(features, dim=1)

    return features


def extract_features(model, dataloader, last_only=False):
    features_all, y_all = [], []
    for batch in tqdm(dataloader):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, mask, y = batch
        # with torch.no_grad():
        with torch.no_grad():
            x, y = x.cuda(), y.detach().cpu()
            if last_only:
                features = extract_features_last_only_single(model, x).detach().cpu()
            else:
                features = extract_features_single(model, x).detach().cpu()
            features_all.append(features)
            y_all.append(y)

    features_all = torch.cat(features_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    return features_all, y_all


def preprocess_meta(df, onehot=True, normalize=True):
    # keep age	sex	body_site	melanoma_history	resolution cols
    df = df.drop(columns=df.columns.difference(['age', 'sex', 'body_site', 'melanoma_history', 'resolution']))

    # Convert to float
    df['age'] = df['age'].apply(lambda x: x.split(':')[0][1:]).astype(float)

    # One-hot encode
    if onehot:
        onehot_features_cols = ['sex', 'body_site', 'melanoma_history']

        onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        onehot_encoder.fit(df[onehot_features_cols].values)
        onehot_encoded = onehot_encoder.transform(df[onehot_features_cols].values).toarray()
        onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(onehot_features_cols))
        df = pd.concat([df, onehot_encoded_df], axis=1)
        df = df.drop(columns=onehot_features_cols)

    # Normalize non-one-hot-encoded features
    if normalize:
        seq_features_cols = ['age', 'resolution']
        for col in seq_features_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df


def extract_features_patches(model, dataloader, patch_size=384, last_only=False):
    features_all, y_all = [], []
    for batch in tqdm(dataloader):
        x, y, _ = batch[0]
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()

            B, C, H, W = x.shape
            P = patch_size

            # Pad image to be divisible by P
            W_pad = (P - x.shape[3] % P) if x.shape[3] % P != 0 else 0
            H_pad = (P - x.shape[2] % P) if x.shape[2] % P != 0 else 0
            x = torch.nn.functional.pad(x, (0, P - W_pad, 0, H_pad), mode='reflect')

            # Extract patches
            x = x \
                .unfold(2, P, P) \
                .unfold(3, P, P)

            # Concat to batch dimension
            # (B, C, H_patches, W_patches, patch_size, patch_size) -> 
            # (B * H_patches * W_patches, C, patch_size, patch_size)
            H_patches, W_patches = x.shape[2:4]
            x = x.reshape(B * H_patches * W_patches, C, P, P)

            # Extract features
            if last_only:
                features = extract_features_last_only_single(model, x)
            else:
                features = extract_features_single(model, x)

            # Reshape back as expected from PatchEmbed
            # (B * H_patches * W_patches, *backbone_out_shape) ->
            # (B, H * W, *backbone_out_shape)
            features = features.reshape(B, H_patches * W_patches, *features.shape[1:])
            
            features_all.append(features.cpu())
            y_all.append(y.cpu())

    features_all = torch.cat(features_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    return features_all, y_all


def heatmap(data, row_labels, col_labels, ax=None,
            plot_cbar=False, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if plot_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar, ax.figure


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def build_cm_heatmap(cm, classes=None):
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()
    if classes is None:
        classes = ['0', '1']

    image, _, figure = heatmap(
        cm, 
        row_labels=classes, 
        col_labels=classes, 
        vmin=0, 
        vmax=cm.sum(1).max(), 
        cmap='YlGn'
    )
    annotate_heatmap(image, valfmt="{x:d}")

    return figure


def duplicated_tensors(series, keep='first'):
    """"Return boolean mask of duplicated tensors in series.
    """
    lens = series.apply(lambda x: np.prod(x.shape))
    hashes = series.apply(
        lambda x: hash(x.flatten().numpy().tobytes())
    )
    return pd.DataFrame(
        {
            'len': lens,
            'hash': hashes,
        }
    ).duplicated(keep=keep)


def deduplicate_repeated_augs(df):
    """
    Deduplicate repeated augmentations DataFrame.

    Somehow repeating random transform on embedding generation 
    ends up with duplicated embeddings. Approach below deduplicate 
    each DataFrame. Given that dataframes (folds) does not 
    intersect, it deduplicates whole dataset.
    """
    duplicated_path = df['path'].duplicated(keep='first').values
    duplicated_label = df['label'].apply(lambda x: x.item()).duplicated(keep='first').values
    duplicated_features = duplicated_tensors(df['features'], keep='first').values
    return df[~(duplicated_features & duplicated_label & duplicated_path)]


def check_no_pairwise_intersection(dfs):
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            assert len(set(dfs[i]['path'].values).intersection(dfs[j]['path'].values)) == 0


def check_unique_pathes_same(df1, df2):
    pathes1 = sorted(list(set(df1['path'].values)))
    pathes2 = sorted(list(set(df2['path'].values)))
    assert pathes1 == pathes2


def load_embeddings(pathes, deduplicate=True):
    dfs = dict()
    for path in pathes:
        logging.info(f'path: {path}')
        with open(path, 'rb') as f:
            df = pickle.load(f)
        logging.info(f'\traw shape: {df.shape}')
        if deduplicate:
            df = deduplicate_repeated_augs(df)
            logging.info(f'\tdeduplicated shape: {df.shape}')
        dfs[path] = df

    # TODO: check_no_pairwise_intersection does not work 
    # if aug + no-aug are used but is not a problem because 
    # no intersections was checked manually and train / val 
    # intersection is checked later.

    # Manual check: set only val / only val_aug pathes
    # in either embedding_pathes or embedding_pathes_aug_with_repeats.

    # check_no_pairwise_intersection(list(dfs.values()))

    return dfs


class PenalizedBinaryFBetaScore(BinaryFBetaScore):
    """
    BinaryFBetaScore with additive penalty. Trying to penalize
    F-beta metric in situations when one or both classes are
    prediced "badly".

    Note: binary stats scores are computed according threshold given
    in initialization.

    Penalty is:
                class              penalty                      hard penalty        condition
                0       1
    quality     good    good       0                            0                   TPR > FNR and TNR > FPR
                good    bad        TPR - FNR                    1                   TPR < FNR and TNR > FPR
                bad     good       TNR - FPR                    1                   TPR > FNR and TNR < FPR
                bad     bad        (TPR - FNR) + (TNR - FPR)    2                   TPR < FNR and TNR < FPR

    Note: output limits are [-2, 1], higher is better.
    """

    def __init__(
        self,
        mode: Literal["soft", "hard"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.mode = mode

    def compute(self) -> Tensor:
        fbeta = super().compute()

        tp, fp, tn, fn = self._final_state()
        
        tnr = _safe_divide(tn, tn + fp)
        fpr = _safe_divide(fp, tn + fp)
        fnr = _safe_divide(fn, tp + fn)
        tpr = _safe_divide(tp, tp + fn)

        p_diff_score = tpr - fnr  # higher is better, limits are [-1, 1]
        n_diff_score = tnr - fpr  # higher is better, limits are [-1, 1]

        p_penalty = 1 if self.mode == "hard" else p_diff_score
        n_penalty = 1 if self.mode == "hard" else n_diff_score

        penalty = (p_penalty if p_diff_score < 0 else 0) + (n_penalty if n_diff_score < 0 else 0)

        return fbeta + penalty


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def build_aggregation(features):
    # features.shape == (1, N_patches, E)
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    features = features.squeeze().astype(np.float32)
    return np.concatenate(
        [
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.max(features, axis=0),
            np.min(features, axis=0),
            np.quantile(features, 0.5, axis=0),
        ],
        axis=0,
    )


def get_X_y_groups(df):
    X = np.array(df['features'].apply(build_aggregation).tolist())
    y = np.array(df['label'].apply(np.array).tolist())
    groups = df['path'].values
    return X, y, groups


class SavePredictionsCallback(Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.predictions = []
        self.filenames = []
    
    def on_predict_batch_end(
        self, 
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.filenames.extend([path.split('/')[-1] for path in batch[-1]])
        self.predictions.append(outputs)

    def on_predict_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        self.predictions = torch.softmax(torch.cat(self.predictions, dim=0), dim=1).cpu().numpy()[:, 1]
        df = pd.DataFrame(
            {
                'filename': self.filenames,
                'relapse': self.predictions,
            }
        )
        df.to_csv(self.save_path, index=False)
        
        self.predictions = []
        self.filenames = []


def oldest_checkpoint(filenames):
    filenames = [filename for filename in filenames if 'epoch' in filename]
    # format is epoch={epoch}-step={step}.ckpt
    # get path with largest step
    return sorted(filenames, key=lambda x: int(x.split('=')[2].split('.')[0]))[-1]
