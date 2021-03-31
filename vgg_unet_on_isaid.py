# %%
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

# %% View Dataset
original_image = "./data/iSAID_patches/train/rgb_source_images/P0000_0_800_0_800.png"
label_image_semantic = "./data/iSAID_patches/train/sem_label_mask_cat/P0000_0_800_0_800.png"

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow(Image.open(original_image))
axs[0].grid(False)

label_image_semantic = Image.open(label_image_semantic)
label_image_semantic = np.asarray(label_image_semantic)
axs[1].imshow(label_image_semantic)
axs[1].grid(False)

#%%
label_image_semantic

# %% Train on training data
epochs = 1
from keras_segmentation.models.unet import vgg_unet

# If utilizing the GPU, this may be needed to prevent OOM errors.
# import tensorflow as tf
# gpus= tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

n_classes = 16 # iSAID Dataset
# See https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py
model = vgg_unet(n_classes=n_classes ,  input_height=800, input_width=800)

model.train( 
    train_images =  "./data/iSAID_patches/train/rgb_source_images",
    train_annotations = "./data/iSAID_patches/train/sem_label_mask_cat",
    checkpoints_path = "./checkpoints/iSAID/vgg_unet" , 
    epochs=epochs, 
    verify_dataset=False,
    batch_size=8,
    steps_per_epoch=3500,
    auto_resume_checkpoint=True
)
# %% Predict
start = time.time()

input_image = "./data/iSAID_patches/train/rgb_source_images/P0000_600_1400_2400_3200.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="out.png"
)
done = time.time()

fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

img_orig = Image.open(input_image)
axs[0].imshow(img_orig)
axs[0].set_title('original image-001.jpg')
axs[0].grid(False)

axs[1].imshow(out)
axs[1].set_title('prediction image-out.png')
axs[1].grid(False)

validation_image = "./data/iSAID_patches/train/sem_label_mask_cat/P0000_600_1400_2400_3200.png"
axs[2].imshow( Image.open(validation_image))
axs[2].set_title('true label image-001.png')
axs[2].grid(False)

elapsed = done - start
# %%
