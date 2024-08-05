from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
import torch.nn.functional as F
import sys; sys.path.append('../')
from models.SERNet_Former import SERNet_Former, SERNetConfig
from transformers import Trainer, TrainingArguments
from utils.losses import dice_loss_binary as dice_loss
from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation
from adan_pytorch import Adan
from utils.initialization import init_weights
from utils.metrics import iou as compute_iou


class CustomTrainer(Trainer):
    def __init__(self, dice_fraction=0.2, ce_fraction=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_fraction = dice_fraction
        self.ce_fraction = ce_fraction

        # self.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        # self.add_callback(TensorBoardCallback())
        # Uncomment the following line if you want to use Weights and Biases
        # self.add_callback(WandbCallback())

    def compute_loss(self, model, inputs, return_outputs=False):
        image = inputs["image"]
        mask = inputs["mask"].contiguous().view(-1).float()
        outputs = model(image)
        logits = outputs.contiguous().view(-1)
        ce_loss = F.cross_entropy(logits, mask)
        dice_loss_ = dice_loss(logits, mask)
        loss = self.ce_fraction*ce_loss+self.dice_fraction*dice_loss_
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = Adan(self.model.parameters(), weight_decay = 0.02)
        return self.optimizer
    
    def compute_metrics(self, p):
        logits = p.predictions
        labels = p.label_ids
        
        iou = compute_iou(logits, labels)
        return {"iou": iou}



    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", n_iters: int | None = 100):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model.eval()
        # Initialize containers to hold logits and labels
        ious = []

        # Iterate through the evaluation dataloader
        
        for i, inputs in enumerate(eval_dataloader):
            # Move inputs to the device
            inputs = self._prepare_inputs(inputs)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs['image'])
                logits = outputs
            
            iou = compute_iou(logits, inputs['mask'])
            ious.append(iou)
            if n_iters is not None and i>n_iters:
                break

        
        iou = np.mean(ious)

        # Log metrics
        metrics = {f"{metric_key_prefix}_iou": iou}
        self.log(metrics)
        model.train()

        return metrics


# Training script
if __name__ == '__main__':

    # Initialize model configuration and model
    config = SERNetConfig(num_labels=1)
    model = SERNet_Former(config)
    model.apply(init_weights)

    # Load dataset
    train_dataset = EasyPortraitDataset(
        IMGS_TRAIN_DIR,
        ANNOTATIONS_TRAIN_DIR,
        get_training_augmentation(),
        binary=True
    )

    eval_dataset = EasyPortraitDataset(
        IMGS_VAL_DIR,
        ANNOTATIONS_VAL_DIR,
        get_val_test_augmentation(),
        binary=True
    )

    # Initialize TrainingArguments
    training_args = TrainingArguments(
        output_dir='../results',
        num_train_epochs=100,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        report_to="none",
        logging_dir='../logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_iou",  # Use mIoU for selecting the best model
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=1  # Keep only the best model
    )

    # Initialize custom trainer with callbacks and custom metric
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add custom metric
        data_collator=SegmentationDataCollator()
    )


    # Start training
    trainer.train()