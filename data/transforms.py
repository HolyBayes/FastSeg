import albumentations as albu
import albumentations.pytorch as albu_pytorch


def post_augmentation():
    post_transforms = [
        albu.Normalize(),
        albu_pytorch.transforms.ToTensorV2()  
    ]
    return albu.Compose(post_transforms)

IMG_SIZE = 512

def get_training_augmentation(img_size=IMG_SIZE):
    train_transforms = [
        albu.ShiftScaleRotate(rotate_limit=25, border_mode=0, p=0.5),
        albu.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0, value=0),
        albu.Resize(height=img_size, width=img_size, p=1),
        albu.GaussNoise(p=0.3),

        albu.OneOf(
            [
                albu.CLAHE(p=0.5),
                albu.RandomBrightnessContrast(p=0.6),
                albu.RandomGamma(p=0.4),
            ],
            p=0.8,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=3, p=0.4),
                albu.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.7,
        ),
        albu.HueSaturationValue(p=0.15),
        post_augmentation()
    ]
    return albu.Compose(train_transforms,
                        additional_targets={'depth': 'mask'})


def get_val_test_augmentation(img_size=IMG_SIZE):
    val_test_transforms = [
        albu.PadIfNeeded(img_size, img_size),
        albu.Resize(height=img_size, width=img_size, p=1),
        post_augmentation()
    ]
    return albu.Compose(val_test_transforms,
                        additional_targets={'depth': 'mask'})