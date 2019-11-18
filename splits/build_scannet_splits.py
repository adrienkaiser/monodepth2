
import os
import random # shuffle()

scannetdatafolder = "/path/to/Scannet"
monodepth2home = "/path/to/monodepth2"

# TODO: Improve this arbitrary train/val separation ?
train_scenes = [
    "scene0220_02", # https://github.com/ScanNet/ScanNet
    "scene0294_02",
    "scene0451_05",
    "scene0567_01",
    "office0", # http://graphics.stanford.edu/projects/bundlefusion/#data
    "office1",
    "office3",
    "offices-rgbd"] # http://graphics.stanford.edu/projects/3dlite/#data
val_scenes = [
    "scene0271_01",
    "apt-rgbd"]

scannetsplitfilesfolder = os.path.join(monodepth2home, "splits/scannet")

if not os.path.exists(scannetsplitfilesfolder):
    print("Creating splits folder:", scannetsplitfilesfolder)
    os.makedirs(scannetsplitfilesfolder)

train_filepaths = []
val_filepaths = []

# Browse scene folders in main ScanNet directory
for scene in os.listdir(scannetdatafolder):
    if not os.path.isdir(os.path.join(scannetdatafolder, scene)):
        continue
    if not scene in train_scenes and not scene in val_scenes:
        continue
    print(scene)
    indexes_int = []
    for file in os.listdir(os.path.join(scannetdatafolder, scene, "depth" if "scene0" in scene else "")):
        if not ".png" in file:
            continue
        frame_index = file.split(".png")[0] if "scene0" in scene else file.split("frame-")[1].split(".depth.png")[0]
        indexes_int.append(int(frame_index))
    indexes_int.sort()
    if scene in train_scenes:
        train_filepaths += [scene + " " + str(id) + " l" for id in indexes_int[1:-1]] # exclude first and last for neighbor frame matching
    elif scene in val_scenes:
        val_filepaths += [scene + " " + str(id) + " l" for id in indexes_int[1:-1]]

random.shuffle(train_filepaths)
random.shuffle(val_filepaths)

train_file = open(os.path.join(scannetsplitfilesfolder, "train_files.txt"), "w")
train_file.write("\n".join(train_filepaths))
train_file.close()

val_file = open(os.path.join(scannetsplitfilesfolder, "val_files.txt"), "w")
val_file.write("\n".join(val_filepaths))
val_file.close()
