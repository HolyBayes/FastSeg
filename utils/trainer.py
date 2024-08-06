from transformers import Trainer
from adan_pytorch import Adan
import sys; sys.path.append('../')
from utils.losses import dice_loss_binary as dice_loss
from utils.metrics import iou as compute_iou
import torch.nn.functional as F
import torch
import numpy as np


class SemanticSegmentationTrainer(Trainer):
    def __init__(self, dice_fraction=0.2, ce_fraction=0.8, label_names=['person'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_fraction = dice_fraction
        self.ce_fraction = ce_fraction
        self.label_names = label_names

        # self.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        # self.add_callback(TensorBoardCallback())
        # Uncomment the following line if you want to use Weights and Biases
        # self.add_callback(WandbCallback())

    def compute_loss(self, model, inputs, return_outputs=False):
        image = inputs["image"]
        mask = inputs["mask"]
        outputs = model(image)
        if mask.shape[-2:] != outputs.shape[-2:]:
            outputs = F.interpolate(outputs, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.contiguous().view(-1).float()
        logits = outputs.contiguous().view(-1)
        ce_loss = F.binary_cross_entropy_with_logits(logits, mask)
        dice_loss_ = dice_loss(logits, mask)
        loss = self.ce_fraction*ce_loss+self.dice_fraction*dice_loss_
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = Adan(self.model.parameters(), weight_decay = 0.02)
        return self.optimizer
    

    def evaluate(self, eval_dataset=None, ignore_keys=False, metric_key_prefix: str = "eval", n_iters: int | None = 100):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        # Initialize containers to hold logits and labels
        ious = []

        # Iterate through the evaluation dataloader
        
        for i, inputs in enumerate(eval_dataloader):
            # Move inputs to the device
            inputs = self._prepare_inputs(inputs)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(inputs['image'])
                
            mask = inputs['mask']
            if mask.shape[-2:] != logits.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

            iou = compute_iou(logits, mask)
            ious.append(iou)
            if n_iters is not None and i>n_iters:
                break

        
        iou = np.mean(ious)

        # Log metrics
        metrics = {f"{metric_key_prefix}_iou": iou}
        self.log(metrics)
        self.model.train()

        return metrics