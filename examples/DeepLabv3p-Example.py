# %% Imports
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# %% Data Wrangling
original_image = "../data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/001.jpg"
label_image_semantic = "../data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/001.png"

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(original_image))
axs[0].grid(False)

label_image_semantic = Image.open(label_image_semantic)
label_image_semantic = np.asarray(label_image_semantic)
axs[1].imshow(label_image_semantic)
axs[1].grid(False)

# %% Define Model
base_model = tf.keras.models.