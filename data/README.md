Datasets can be downloaded from:

- [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)
- [iSAID](https://captain-whu.github.io/iSAID/dataset.html)

Once downloaded folder structure should look like this:

```
./data
    aerialSemSegDroneDataset
        datafiles
    iSAID_patches
        test
        train
        val
```

Note: The iSAID dataset is built upon the authors previous dataset, [DOTA-v1.5](https://captain-whu.github.io/DOTA/dataset.html), and requires the download of the original images from that dataset. The raw original images and ground truth label images must be preprocessed via the iSAID instructions found in the [iSAID_Devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit) to generate the iSAID_patches files.