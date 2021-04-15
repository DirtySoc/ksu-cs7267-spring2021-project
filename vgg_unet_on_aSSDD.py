# Source code: https://www.kaggle.com/bulentsiyah/deep-learning-based-semantic-segmentation-keras/notebook
# Dataset: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset
# %% Imports
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json

# %% View dataset image
original_image = "./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/001.jpg"
label_image_semantic = "./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/001.png"

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(original_image))
axs[0].grid(False)

label_image_semantic = Image.open(label_image_semantic)
axs[1].imshow(label_image_semantic)
axs[1].grid(False)
# %% Build model.
# Models can be found at https://github.com/divamgupta/image-segmentation-keras.
from keras_segmentation.models.unet import vgg_unet

epochs = 100
n_classes = 23
model = vgg_unet(n_classes=n_classes, input_height=416, input_width=608)

# %% Train model. Rerun to continue trainig for 5 more epochs.
model.train(
    train_images="./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/",
    train_annotations="./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/",
    checkpoints_path="./checkpoints/aerialSemSegDroneDataset/vgg16_unet/ckpt",
    epochs=epochs,
    verify_dataset=False
    )

history = model.history
with open("figures/vgg16_train_metrics.json", "w") as outfile:
    json.dump(history.history, outfile)

# %% Define func to gen predictions
def generate_prediction(img):
    input_image = "./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/" + img + ".jpg"
    start = time.time()
    out = model.predict_segmentation(
        inp=input_image,
        out_fname="tmp/out.png"
    )
    done = time.time()

    fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

    img_orig = Image.open(input_image)
    axs[0].imshow(img_orig)
    axs[0].set_title('original image-' + img)
    axs[0].grid(False)

    axs[1].imshow(out)
    axs[1].set_title('prediction image-out.png')
    axs[1].grid(False)

    validation_image = "./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/" + img + ".png"
    axs[2].imshow( Image.open(validation_image))
    axs[2].set_title('true label image-' + img)
    axs[2].grid(False)

    elapsed = done - start
    print(elapsed)

# %% Explore Results
generate_prediction("004")
generate_prediction("005")
generate_prediction("006")

# %% Evaluation Metrics
inp_images_dir='./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/original_images/'
annotations_dir='./data/aerialSemSegDroneDataset/dataset/semantic_drone_dataset/label_images_semantic/'
evaluation = model.evaluate_segmentation( inp_images_dir=inp_images_dir  , annotations_dir=annotations_dir )
print(evaluation)
evaluation['class_wise_IU'] = evaluation['class_wise_IU'].tolist()

with open("figures/vgg16_test_eval.json", "w") as outfile:
    json.dump(evaluation, outfile)

# %% Plot Accuracy over epoch
plt.plot(history.history['accuracy'],  label='Train Accuracy')
plt.title('VGG16 Unet Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# %% Plot Loss over epoch
plt.plot(history.history['loss'],  label='Train Loss')
plt.title('VGG16 Unet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# %% 
