import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CityScapesDataset(Dataset):
    def __init__(self, images_dir , labels_dir, transform_img=None , transform_label=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.image_filenames = sorted(os.listdir(images_dir))
        self.labels_filenames= sorted(os.listdir(labels_dir)) 

        self.transform_img = transform_img or transforms.Compose([transforms.Resize((256,512)),
        transforms.ToTensor()])

        self.transform_label= transform_label or transforms.Compose([transforms.Resize((256,512)),
        transforms.ToTensor()])

    def __len__(self):
        """return the number of images in the dataset"""
        return len(self.image_filenames)

    def __getitem__(self ,idx):
        img_path=os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path=os.path.join(self.labels_dir, self.labels_filenames[idx])

        image=Image.open(img_path).convert("RGB")
        mask=Image.open(mask_path).convert("L")

        image_tensor=self.transform_img(image)
        mask_tensor=self.transform_label(mask)

        mask_tensor=mask_tensor.squeeze(0).long()

        return image_tensor, mask_tensor    


class CityscapesDataLoader:
    def __init__(self ,dataset, batch_size=4 , shuffle=True , num_workers=2):
        self.dataloader=DataLoader(dataset=dataset ,
         batch_size=batch_size , shuffle=shuffle , num_workers=num_workers)

    def get_loader(self):
        return self.dataloader    


IMAGE_TRAIN_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\train\img"
IMAGE_VAL_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\val\img"
LABEL_TRAIN_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\train\label"
LABEL_VAL_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\val\label"        



train_dataset=CityScapesDataset(IMAGE_TRAIN_DIR , LABEL_TRAIN_DIR)
val_dataset=CityScapesDataset(IMAGE_VAL_DIR , LABEL_VAL_DIR)


train_dataloader=CityscapesDataLoader(train_dataset , batch_size=4 , shuffle=True , num_workers=2)
val_dataloader=CityscapesDataLoader(val_dataset , batch_size=4 , shuffle=False , num_workers=2)