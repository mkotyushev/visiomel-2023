import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

from src.model.visiomel_model import VisiomelModel
from src.model.quadtree_embedding import QuadtreeEmbedding


class QuadtreeClassifier(VisiomelModel):
    def __init__(
        self, 
        backbone_name, 
        splitter_hidden_size=64, 
        patch_size=64,
        num_classes: int = 2, 
        optimizer_init: Optional[Dict[str, Any]] = None,
        lr_scheduler_init: Optional[Dict[str, Any]] = None,
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        finetuning: Optional[Dict[str, Any]] = None,
        log_norm_verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embedding = QuadtreeEmbedding(
            backbone, 
            splitter_hidden_size=splitter_hidden_size, 
            patch_size=patch_size
        )
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x, random_split=False):
        preds, split_decision_logits = [], []
        for i in range(x.shape[0]):
            pred, logits = self.embedding(
                x[i, ...].unsqueeze(0), 
                output_split_logits=True, 
                random_split=random_split
            )
            preds.append(pred)
            split_decision_logits.append(logits)

        x = torch.cat(preds, 0)
        split_decision_logits = torch.cat(split_decision_logits, 0)

        x = x.mean((2, 3))
        x = self.classifier(x)

        return x, split_decision_logits
    
    def compute_loss_preds(self, batch, random_split=False):
        x, y = batch
        preds, split_decision_logits = self(x, random_split=False)
        loss = F.cross_entropy(preds, y)
        
        if not random_split:
            return loss, {'ce': loss}, preds
        
        preds_random_split, random_split_decisions = self(x, random_split=True)
        loss_random_split = F.cross_entropy(preds_random_split, y)

        loss_split_decision = F.cross_entropy(
            split_decision_logits.flatten(end_dim=-2), 
            random_split_decisions.flatten()
        )
        apply_split_decision_loss = ((torch.sign(loss_random_split - loss) + 1) / 2).detach()
        loss_split_decision = loss_split_decision * apply_split_decision_loss

        return loss + loss_split_decision, {
            'ce': loss, 
            'sd': loss_split_decision
        }, preds
    
    def training_step(self, batch, batch_idx, **kwargs):
        return super().training_step(batch, batch_idx, random_split=True)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None, **kwargs):
        return super().validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, random_split=False)