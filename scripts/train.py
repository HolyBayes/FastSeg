from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
import torch.nn.functional as F
import sys; sys.path.append('../')
from models.SERNet_Former import SERNet_Former, SERNetConfig
from transformers import Trainer, TrainingArguments
from utils.losses import dice_loss_binary as dice_loss
from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation

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
        mask = inputs["mask"].contiguous().view(-1)
        outputs = model(image)
        logits = outputs.contiguous().view(-1)
        ce_loss = F.cross_entropy(logits, mask.float())
        dice_loss_ = dice_loss(logits, mask)
        loss = self.ce_fraction*ce_loss+self.dice_fraction*dice_loss_
        return (loss, outputs) if return_outputs else loss

# Training script
if __name__ == '__main__':

    # Initialize model configuration and model
    config = SERNetConfig(num_labels=1)
    model = SERNet_Former(config)

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
        output_dir='./results',
        num_train_epochs=100,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        report_to="none",
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",  # Use mIoU for selecting the best model
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


    # import pdb; pdb.set_trace()
    # Start training
    # print(eval_dataset[0])
    # print(len(eval_dataset))
    trainer.train()