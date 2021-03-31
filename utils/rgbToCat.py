# %% This converts RGB sem seg masks to a new image where each pixel 
#    is the value of it's category
# MUST BE RUN FROM INSIDE UTILS DIR

# %% Import and view data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

# %% 

# Define directories
source_image_dir = '../data/iSAID_patches/train/images'
rgb_source_image_Dir = '../data/iSAID_patches/train/rgb_source_images'
instance_id_rgb_Dir = '../data/iSAID_patches/train/inst_sem_mask'
instance_color_rgb_Dir = '../data/iSAID_patches/train/sem_label_mask'
sem_label_mask_cat_Dir = '../data/iSAID_patches/train/sem_label_mask_cat'
try:
    os.mkdir(instance_id_rgb_Dir)
except FileExistsError:
    pass
try:
    os.mkdir(instance_color_rgb_Dir)
except FileExistsError:
    pass
try:
    os.mkdir(sem_label_mask_cat_Dir)
except FileExistsError:
    print("output dir for category labeled images exists... overwriting images...")

for filepath in glob.iglob(os.path.join(source_image_dir, "*_instance_id_RGB.png")):
    filename = filepath.split('\\')[-1] + '.png'
    filename = filename.split('_instance')[0] + '.png'
    os.rename(filepath, os.path.join(instance_id_rgb_Dir, filename))

for filepath in glob.iglob(os.path.join(source_image_dir, "*_instance_color_RGB.png")):
    filename = filepath.split('\\')[-1] + '.png'
    filename = filename.split('_instance')[0] + '.png'
    os.rename(filepath, os.path.join(instance_color_rgb_Dir, filename))

os.rename(source_image_dir, rgb_source_image_Dir)

# %% Verfiy iSAID training patches are in seperate directories.
originalRgbImg = '../data/iSAID_patches/train/rgb_source_images/P0000_0_800_0_800.png'
originalSemMask = '../data/iSAID_patches/train/sem_label_mask/P0000_0_800_0_800.png'

fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

axs[0].imshow( Image.open(originalRgbImg))
axs[0].grid(False)

originalRgbImg = Image.open(originalSemMask)
# originalRgbImg = np.asarray(originalSemMask)
axs[1].imshow(originalRgbImg)
axs[1].grid(False)

# %% Define mapping between color and category.
gsMapping = { 0: 0,   # background
             29: 1,   # swimming pools
              7: 2,   # ships
             37: 3,   # baseball diamonds
             44: 4,   # storage tanks
             14: 5,   # small vehicle
             51: 6,   # tennis court
             82: 7,   # bridge
             96: 8,   # soccer field
             66: 9,   # ground track field
             76: 10,  # harbor
             89: 11,  # large vehicle
            104: 12,  # plane
            127: 13,  # roundabout
             59: 14,  # basketball court
             22: 15}  # helicopter

def getClass(x):
    return gsMapping[x] 

getClass_v = np.vectorize(getClass)

# %% Process sem seg maps into labeled pixel maps.
isaidSemMaskPatches = '../data/iSAID_patches/train/sem_label_mask/*.png'

print("Create output directory. Writing labeled pixel maps now...")
for filepath in glob.iglob(isaidSemMaskPatches):
    filename = filepath.split('\\')[-1] # May not work on unix systems....
    img = Image.open(filepath).convert('L')  # convert to grayscale with pillow
    img = np.asarray(img)
    img = Image.fromarray(getClass_v(img))   # convert to category scale
    img.save(os.path.join(sem_label_mask_cat_Dir, filename))

# %%
