#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


DATA_DIR = "/data01/database/Skin"
pix_size=256
img_height=3888//pix_size
img_width=2592//pix_size


# In[3]:


x_train_dir = os.path.join(DATA_DIR+"/images")
y_train_dir = os.path.join(DATA_DIR+"/masks")

x_valid_dir = os.path.join(DATA_DIR+"/val_images")
y_valid_dir = os.path.join(DATA_DIR+"/val_masks")

x_test_dir = os.path.join(DATA_DIR+"/test_images")
y_test_dir = os.path.join(DATA_DIR+"/test_masks")


# In[4]:


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# In[5]:


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


# In[6]:


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['trouble']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        
        #image = Image.open(self.images_fps[i]).convert("RGB")
        #image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.open(self.images_fps[i]).convert('RGB')
        image = np.array(image)
        # num_img : numpy.ndarray
        #mask = cv2.imread(self.masks_fps[i], 0)
        mask = Image.open(self.masks_fps[i])
        mask = np.array(mask)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


# In[7]:


# Lets look at data we have

dataset = Dataset(x_train_dir, y_train_dir, classes=['trouble'])

image, mask = dataset[84] # get some sample
visualize(
    image=image, 
    skin_mask=mask.squeeze(),
)


# In[8]:


import albumentations as albu


# In[9]:


from albumentations.core.transforms_interface import ImageOnlyTransform

import numpy as np

from albumentations import HorizontalFlip

class AugMix(ImageOnlyTransform):
    """Augmentations mix to Improve Robustness and Uncertainty.
    Args:
        image (np.ndarray): Raw input image of shape (h, w, c)
        severity (int): Severity of underlying augmentation operators.
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly
          from [1, 3]
        alpha (float): Probability coefficient for Beta and Dirichlet distributions.
        augmentations (list of augmentations): Augmentations that need to mix and perform.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[HorizontalFlip()], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")


# In[10]:


def get_training_augmentation():
    augs = [albu.HorizontalFlip(always_apply=True),
        albu.Blur(always_apply=True),
        albu.OneOf(
        [albu.ShiftScaleRotate(always_apply=True),
        albu.GaussNoise(always_apply=True),]
        ),
        albu.Cutout(always_apply=True),
        albu.IAAPiecewiseAffine(always_apply=True),
        albu.RandomBrightnessContrast(always_apply=True),]

    return albu.Compose([
        AugMix(width=3, depth=2, alpha=.2, p=1., augmentations=augs),
    ])


# def get_training_augmentation():
#     train_transform = [

#         albu.HorizontalFlip(p=0.5),

#         albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        
#         albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
#         albu.RandomCrop(height=256, width=256, always_apply=True),

#         albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.IAAPerspective(p=0.5),

#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),
#         albu.OneOf(
#             [
#                 albu.ShiftScaleRotate(always_apply=True),
#                 albu.GaussNoise(always_apply=True),
#                 albu.RandomBrightness(limit=0.2, always_apply=True),
#             ]
#         ),

#         albu.OneOf(
#             [
#                 albu.IAASharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1),
#                 albu.HueSaturationValue(p=1),
                
#             ],
#             p=0.9,
#         ),
#     ]
#     return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# In[11]:


#### Visualize resulted augmented images and masks

augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=['trouble'],
)

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[73]
    visualize(image=image, mask=mask.squeeze(-1))


# In[12]:


import torch
import numpy as np
import segmentation_models_pytorch as smp


# In[13]:


ENCODER = 'se_resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['trouble']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# In[14]:


train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)


# In[15]:


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
# ce_loss = torch.nn.CrossEntropyLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# In[16]:



# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# In[17]:


# train model for 40 epochs

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    #valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    #if max_score < valid_logs['iou_score']:
    #    max_score = valid_logs['iou_score']
    torch.save(model, '/data01/notebooks/jh/skin/test_model.pth')
    #    print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# In[18]:


# load best saved checkpoint
test_model = torch.load('/data01/notebooks/jh/skin/test_model.pth')


# In[19]:


# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# In[20]:


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=test_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)


# In[21]:


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)


# In[40]:


src = np.full((pix_size*img_height, pix_size*img_width, 3), (0, 0, 0), dtype=np.uint8)
n=0
for i in range(img_width):
    for j in range(img_height):

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
    
        gt_mask = gt_mask.squeeze()
    
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = test_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    


        pr_mask2 = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)*150
        img = cv2.addWeighted(image_vis, 0.7, pr_mask2.astype(np.uint8), 0.3, 0)
        src[j*pix_size:(j+1)*pix_size,i*pix_size:(i+1)*pix_size]=img
        n=n+1

        
plt.figure(figsize=(15, 10))
plt.subplot(1, 1,  1)
plt.xticks([])
plt.yticks([])
plt.imshow(src)

   


# In[39]:


for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = test_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    


    pr_mask2 = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)*150
    img = cv2.addWeighted(image_vis, 0.7, pr_mask2.astype(np.uint8), 0.3, 0)

    

    
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask

    )


# In[24]:


pr_mask


# In[25]:


np.unique(pr_mask)


# In[26]:


n = np.random.choice(len(test_dataset))
   
image_vis = test_dataset_vis[n][0].astype('uint8')
image, gt_mask = test_dataset[n]
    
gt_mask = gt_mask.squeeze()
    
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
pr_mask = test_model.predict(x_tensor)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())


pr_mask2 = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)*100
img = cv2.addWeighted(image_vis, 0.7, pr_mask2.astype(np.uint8), 0.3, 0)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(image_vis)
ax2 = fig.add_subplot(1,3, 2)
ax2.imshow(pr_mask)

ax3 = fig.add_subplot(1,3, 3)
ax3.imshow(img)


# In[27]:


kernel = np.ones((5, 5), np.uint8)
#test = cv2.erode(gt_mask, kernel, interations=1)
plt.imshow(test)
plt.savefig('./test.png')


# In[ ]:


print(pr_mask.shape,image_vis.shape)
#pr_mask_rgb = cv2.cvtColor(pr_mask,cv2.COLOR_GRAY2BGR)
#img = cv2.addWeighted(pr_mask_rgb,0.3,image_vis,0.7,0)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(image_vis)
ax2 = fig.add_subplot(1,3, 2)
ax2.imshow(pr_mask, cmap='gray')
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(image_vis, alpha=1, cmap='gray')
ax3.imshow(pr_mask, alpha=0.2, cmap='gray')


# In[ ]:




