from transformers import TrainingArguments
import sys; sys.path.append('../')
from models.SERNet_Former.model import SERNet_Former, SERNetConfig
from transformers import TrainingArguments

from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation

from utils.initialization import init_weights
from utils.trainer import SemanticSegmentationTrainer



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
        output_dir=f'../results/{model.__class__.__name__}',
        num_train_epochs=100,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        report_to="none",
        logging_dir=f'../logs/{model.__class__.__name__}',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_iou",
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=1  # Keep only the best model
    )

    # Initialize custom trainer with callbacks and custom metric
    trainer = SemanticSegmentationTrainer(
        model=model,
        label_names=['person'],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add custom metric
        data_collator=SegmentationDataCollator()
    )


    # Start training
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)