# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# originalSemMask = 'data\\aerialSemSegDroneDataset\\RGB_color_image_masks\\RGB_color_image_masks\\000.png'
# originalRgbImg = 'data\\aerialSemSegDroneDataset\\dataset\\semantic_drone_dataset\\original_images\\000.jpg'
originalSemMask = 'data\\iSAID_patches\\train\\sem_label_mask\\P0000_0_800_0_800_instance_color_RGB.png'
originalRgbImg = 'data\\iSAID_patches\\train\\rgb_source_images\\P0000_0_800_0_800.png'

print(Image.open(originalSemMask))

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(originalSemMask))
axs[0].grid(False)

originalRgbImg = Image.open(originalRgbImg)
originalRgbImg = np.asarray(originalRgbImg)
axs[1].imshow(originalRgbImg)
axs[1].grid(False)

# %%
import glob

isaidPatches = 'data\\iSAID_patches\\train\\sem_label_mask\\*.png'
outputDir = 'data\\iSAID_patches\\train\\sem_label_mask_gs'
for filepath in glob.iglob(isaidPatches):
    img = Image.open(filepath).convert('L')
    img.save(filepath + '.gs.png')
    print(filepath)

# %%
