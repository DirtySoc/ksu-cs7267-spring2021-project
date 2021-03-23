# Source code: https://www.kaggle.com/bulentsiyah/deep-learning-based-semantic-segmentation-keras/notebook
# Dataset: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset
# %% Imports
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# %% View dataset image
original_image = "./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/001.jpg"
label_image_semantic = "./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/001.png"

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(original_image))
axs[0].grid(False)
``
label_image_semantic = Image.open(label_image_semantic)
label_image_semantic = np.asarray(label_image_semantic)
axs[1].imshow(label_image_semantic)
axs[1].grid(False)
# %% Build model.
# Models can be found at https://github.com/divamgupta/image-segmentation-keras.
from keras_segmentation.models.unet import vgg_unet

epochs = 5
n_classes = 23
model = vgg_unet(n_classes=n_classes, input_height=416, input_width=608)

# %% Train model. Rerun to continue trainig for 5 more epochs.
model.train(
    train_images="./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/",
    train_annotations="./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/",
    checkpoints_path="vgg_unet",
    epochs=epochs
    )
# %% Explore Results

input_image = "./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/001.jpg"
start = time.time()
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

validation_image = "./aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/001.png"
axs[2].imshow( Image.open(validation_image))
axs[2].set_title('true label image-001.png')
axs[2].grid(False)

elapsed = done - start
print(elapsed)

# %% Evaluation Metrics
