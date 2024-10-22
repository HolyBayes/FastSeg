from transformers import TrainingArguments

import sys; sys.path.append('../')
from itertools import chain

from models.SERSegFormer.model import SersegformerForSemanticSegmentation
from models.SERSegFormer.config import SersegformerConfig

from data.dataset import *
from data.transforms import get_training_augmentation, get_val_test_augmentation

from utils.trainer import SemanticSegmentationTrainer
from utils.initialization import init_weights



# Training script
if __name__ == '__main__':

    config = SersegformerConfig(num_labels=1,
                          image_size=512,
                          abg=False,
                          # decoder_hidden_size=128, # Channels compression
                          upsample=True)
    # Initialize model configuration and model
    model = SersegformerForSemanticSegmentation(config) # 25 ms inference time
    model.apply(init_weights)
    
    from safetensors.torch import load_file
    #state_dict = load_file('../checkpoints/segformer-b0/model.safetensors') # finetune Segformer-b0
    #state_dict.pop('decode_head.classifier.weight'); state_dict.pop('decode_head.classifier.bias')
    # rename_fields "segformer' -> "sersegformer"
    #state_dict = {k.replace('segformer', 'sersegformer'):v for k,v in state_dict.items()}

    state_dict = load_file('../checkpoints/serseg/model.safetensors')
    state_dict = {k.replace('upsample.0', 'upsample1').replace('upsample.0', 'upsample2'):v for k,v in state_dict.items()}

    

    model.load_state_dict(state_dict, strict=False)

    
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
        # report_to="none", # Uncomment to diable WandB and Comet
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
        #params=chain(model.decode_head.upsample.parameters(),model.decode_head.classifier.parameters()),
        label_names=['person'],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add custom metric
        data_collator=SegmentationDataCollator()
    )


    # Start training
    trainer.train()
    #trainer.train(resume_from_checkpoint=True)
    
