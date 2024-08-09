from transformers import TrainingArguments
import sys; sys.path.append('../')
from models.SegFormer.model import SegformerForSemanticSegmentation
from models.SegFormer.config import SegformerConfig

from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation

from utils.trainer import SemanticSegmentationTrainer
from safetensors.torch import load_file



# Training script
if __name__ == '__main__':

    config = SegformerConfig(num_labels=1, image_size=512)
    # Initialize model configuration and model
    model = SegformerForSemanticSegmentation(config) # 12 ms inference time
    # state_dict = torch.load('../checkpoints/segformer_b0_cityscapes.pth')
    # state_dict.pop('decode_head.classifier.weight'); state_dict.pop('decode_head.classifier.bias')
    # model.load_state_dict(state_dict, strict=False)
    state_dict = load_file('../checkpoints/segformer-b0/model.safetensors')
    model.load_state_dict(state_dict)


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_params}')
    # Inference 24ms
    
    
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
        metric_for_best_model="eval_iou",  # Use mIoU for selecting the best model
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
    # trainer.train()
    # trainer.evaluate(print_metrics=True)
    # trainer.train(resume_from_checkpoint=True)
    