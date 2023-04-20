import logging
from typing import Any, Dict, Optional
from torch import Tensor
from torch.nn import CrossEntropyLoss

from src.model.visiomel_model import VisiomelModel
from utils.utils import build_classifier


logger = logging.getLogger(__name__)


class SwinTransformerV2Classifier(VisiomelModel):
    def __init__(
        self, 
        model_name: str, 
        num_classes: int = 2, 
        img_size = 224, 
        patch_size: int = 4,
        patch_embed_backbone_name: Optional[str] = None,
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        quadtree: bool = False
    ):
        super().__init__(
            num_classes=num_classes, 
            optimizer_init=optimizer_init, 
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning, 
            pretrained=pretrained,
            log_norm_verbose=log_norm_verbose,
        )
        self.save_hyperparameters()

        if pretrained and patch_size != 4:
            logger.warning(
                f'You are using pretrained model with patch_size={patch_size}. '
                'Pretrained model is trained with patch_size=4. '
                'This will result in resetting of lower layers of the model. '
            )
        
        self.model = build_classifier(
            model_name, 
            num_classes, 
            patch_embed_backbone_name=patch_embed_backbone_name, 
            img_size=img_size, 
            patch_size=patch_size,
            pretrained=pretrained,
            quadtree=quadtree
        )

        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, {'ce': loss}, preds
