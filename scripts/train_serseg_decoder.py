from transformers import TrainingArguments
import sys; sys.path.append('../')
from models.SERSegFormer.model import SersegformerForSemanticSegmentation
from models.SERSegFormer.config import SersegformerConfig

from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation

from utils.trainer import SemanticSegmentationTrainer
from utils.initialization import init_weights

from transformers import get_scheduler, AdamW



# Training script
if __name__ == '__main__':

    config = SersegformerConfig(num_labels=1,
                          image_size=512,
                          abg=False,
                          dbn=False,
                          # decoder_hidden_size=128, # Channels compression
                          upsample=True)
    # Initialize model configuration and model
    # model = SersegformerForSemanticSegmentation(config) # 25 ms inference time
    # model.apply(init_weights)
    
    # from safetensors.torch import load_file
    # state_dict = load_file('../checkpoints/segformer-b0/model.safetensors') # finetune Segformer-b0
    # state_dict.pop('decode_head.classifier.weight'); state_dict.pop('decode_head.classifier.bias')
    
    # # rename_fields "segformer' -> "sersegformer"
    # state_dict = {k.replace('segformer', 'sersegformer'):v for k,v in state_dict.items()}

    # model.load_state_dict(state_dict, strict=True)

    model = SersegformerForSemanticSegmentation.from_pretrained('../checkpoints/serseg_w_decoder/')
    # model = SersegformerForSemanticSegmentation.from_pretrained('../results/SersegformerForSemanticSegmentation_decoder/checkpoint-1500/')

    
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

    exp_name = f"{model.__class__.__name__}_decoder"

    # Initialize TrainingArguments
    training_args = TrainingArguments(
        output_dir=f'../results/{exp_name}',
        num_train_epochs=30,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
#        report_to="none", # Uncomment to diable WandB and Comet
        logging_dir=f'../logs/{exp_name}',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_iou",  # Use mIoU for selecting the best model
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=1  # Keep only the best model
    )

    optimizer = AdamW(model.parameters(), lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    lr_scheduler = get_scheduler(
        name="polynomial",  # or 'cosine', 'polynomial', etc.
        optimizer=optimizer,
        num_warmup_steps=1500,
        scheduler_specific_kwargs={'power': 1.0},
        num_training_steps=training_args.num_train_epochs * len(train_dataset) // training_args.per_device_train_batch_size,
    )

    from itertools import chain

    # Initialize custom trainer with callbacks and custom metric
    trainer = SemanticSegmentationTrainer(
        model=model,
        label_names=['person'],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add custom metric
        data_collator=SegmentationDataCollator(),
        optimizers=(optimizer, lr_scheduler)
    )


    # Start training
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    
