

import logging
from typing import Any, Dict, Optional, Union
from torch import Tensor
from torch.nn import CrossEntropyLoss

from src.model.visiomel_model import VisiomelModel
from utils.utils import build_model
from src.model.drloc.losses import cal_selfsupervised_loss
from lib.SwinTransformer.models.simmim import SwinTransformerV2ForSimMIM


logger = logging.getLogger(__name__)


class SwinTransformerV2SimMIM(VisiomelModel):
    def __init__(
        self, 
        model_name: str, 
        num_classes: int = 2, 
        img_size = 224, 
        patch_size: int = 4,
        patch_embed_backbone_name: Optional[str] = None,
        patch_embed_backbone_pretrained: bool = True,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        quadtree: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        grad_checkpointing: bool = False,
        drloc_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            optimizer_init=optimizer_init, 
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning, 
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
        )
        self.save_hyperparameters()

        if pretrained and patch_size != 4:
            logger.warning(
                f'You are using pretrained model with patch_size={patch_size}. '
                'Pretrained model is trained with patch_size=4. '
                'This will result in resetting of lower layers of the model. '
            )
        
        self.model = build_model(
            model_name, 
            mock_class=SwinTransformerV2ForSimMIM,
            num_classes=0, 
            patch_embed_backbone_name=patch_embed_backbone_name, 
            patch_embed_backbone_pretrained=patch_embed_backbone_pretrained,
            img_size=img_size, 
            patch_size=patch_size,
            pretrained=pretrained,
            quadtree=quadtree,
            grad_checkpointing=grad_checkpointing,
            drloc_params=drloc_params,
        )

        # TODO: called in each VisiomelModel subclass but after subclass __init__
        # need to move to VisiomelModel somehow
        self.unfreeze_only_selected()

    def configure_metrics(self):
        """Configure task-specific metrics."""
        return

    def update_train_metrics(self, preds, batch):
        """Update train metrics."""
        return

    def update_val_metrics(self, preds, batch, dataloader_idx=0):
        """Update val metrics."""
        return

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        img, mask = batch
        loss = self.model(img, mask)
        return loss, {'simmim': loss}, None
