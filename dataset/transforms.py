import cv2
import albumentations as A

def get_transforms(size):
    transforms_train = A.Compose([
        A.RandomResizedCrop(size,size, scale=(0.9, 1), p=1, interpolation=cv2.INTER_LANCZOS4), 
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.2),
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(),
    ])

    transforms_val = A.Compose([
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize()
    ])
    return transforms_train, transforms_val

