import timm
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union

from src.model.visiomel_model import VisiomelClassifier


class TimmClassifier(VisiomelClassifier):
    def __init__(
        self, 
        backbone_name, 
        num_classes: int = 2, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
        lr_layer_decay: Union[float, Dict[str, float]] = 1.0,
        grad_checkpointing: bool = False,
        label_smoothing: float = 0.0,
        skip_nan: bool = False,
    ):
        super().__init__(
            optimizer_init=optimizer_init, 
            lr_scheduler_init=lr_scheduler_init,
            pl_lrs_cfg=pl_lrs_cfg,
            finetuning=finetuning, 
            log_norm_verbose=log_norm_verbose,
            lr_layer_decay=lr_layer_decay,
            label_smoothing=label_smoothing,
            skip_nan=skip_nan,
        )
        self.save_hyperparameters()

        self.classifier = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        self.classifier.set_grad_checkpointing(grad_checkpointing)

        # TODO: called in each VisiomelModel subclass but after subclass __init__
        # need to move to VisiomelModel somehow
        self.unfreeze_only_selected()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def compute_loss_preds(self, batch, *args, **kwargs):
        self.check_batch_dims(batch)

        x, y = batch
        out = self(x)

        y_no_nan, out_no_nan = self.remove_nans(y, out)
        loss = self.loss_fn(out_no_nan, y_no_nan)

        return loss, {'ce': loss}, out
