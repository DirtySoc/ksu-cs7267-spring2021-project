# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% Import data from json file
import json
with open('figures/vgg16_train_metrics.json', 'r') as myfile:
  data = myfile.read()

obj = json.loads(data)

# %% Plot from data in json and save.
import matplotlib.pyplot as plt

plt.plot(obj['accuracy'],  label='Train Accuracy')
plt.title('VGG16 Unet Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.savefig('figures/tmp.jpg')

# %% Plot from data in json and save.
plt.plot(obj['loss'],  label='Train Loss')
plt.title('VGG16 Unet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('figures/tmp.jpg')

# %%
