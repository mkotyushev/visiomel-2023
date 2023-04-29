# Remove PIL image size limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import logging
import timm
import torch
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchvision.datasets.folder import default_loader
from typing import Dict, Optional, Union
from model.patch_embed_with_backbone import SwinTransformerV2Modded
from torchmetrics import Metric
from mock import patch
from sklearn.metrics import log_loss
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


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
        self.add_state("preds", default=[])
        self.add_state("target", default=[])

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        return torch.stack((1 - preds, preds), dim=1), target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape[0] == target.shape[0]

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        self.preds = torch.softmax(torch.cat(self.preds, dim=0), dim=1)
        self.target = torch.cat(self.target, dim=0)
        return log_loss(self.target.cpu().numpy(), self.preds.cpu().numpy(), eps=1e-16).item()


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