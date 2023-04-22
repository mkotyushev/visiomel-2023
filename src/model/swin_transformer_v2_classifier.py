import logging
from typing import Any, Dict, Optional, Union
from torch import Tensor
from torch.nn import CrossEntropyLoss

from src.model.visiomel_model import VisiomelClassifier
from utils.utils import build_model
from src.model.drloc.losses import cal_selfsupervised_loss


logger = logging.getLogger(__name__)


class SwinTransformerV2Classifier(VisiomelClassifier):
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
            num_classes, 
            patch_embed_backbone_name=patch_embed_backbone_name, 
            patch_embed_backbone_pretrained=patch_embed_backbone_pretrained,
            img_size=img_size, 
            patch_size=patch_size,
            pretrained=pretrained,
            quadtree=quadtree,
            grad_checkpointing=grad_checkpointing,
            drloc_params=drloc_params,
        )

        self.loss_fn = CrossEntropyLoss()
        if self.hparams.drloc_params is not None:
            self.criterion_ssup = cal_selfsupervised_loss

        # TODO: called in each VisiomelModel subclass but after subclass __init__
        # need to move to VisiomelModel somehow
        self.unfreeze_only_selected()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        x, y = batch
        out = self(x)
        
        if self.hparams.drloc_params is None:
            loss = self.loss_fn(out, y)
            return loss, {'ce': loss}, out
        else:
            loss_sup = self.loss_fn(out.sup, y)
            loss_ssup, ssup_items = self.criterion_ssup(
                out, 
                self.hparams.drloc_params['drloc_mode'], 
                self.hparams.drloc_params['lambda_drloc']
            )
            loss = loss_sup + loss_ssup
            return loss, {'ce': loss_sup, 'drloc': loss_ssup}, out.sup
