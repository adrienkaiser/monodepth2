
import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class ScannetDataset(MonoDataset):
    """ScannetDataset dataset which loads the original depth maps for ground truth
    """

    # class attributes
    fileformat_map = {
        "scannet" : ["depth/{}.png", "color/{}.jpg", "pose/{}.txt"],
        "bundlefusion" : ["frame-{:06d}.depth.png", "frame-{:06d}.color.jpg", "frame-{:06d}.pose.txt"]
        # add some scenes from 3DLite http://graphics.stanford.edu/projects/3dlite/#data
        }

    def __init__(self, *args, **kwargs):
        super(ScannetDataset, self).__init__(*args, **kwargs)

        Kpx = np.array([[577.870605, 0, 319.5],
                        [0, 577.870605, 239.5],
                        [0, 0, 1]], dtype=np.float32)
        # from ScanNet scene0220_02/intrinsic/intrinsic_depth.txt
        # TODO: load from file ?

        self.full_res_shape = (640, 480)

        self.K = np.array([[Kpx[0,0] / self.full_res_shape[0], 0, Kpx[0,2] / self.full_res_shape[0], 0],
                           [0, Kpx[1,1] / self.full_res_shape[1], Kpx[1,2] / self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(
            self.data_path,
            scene_name,
            ScannetDataset.fileformat_map["scannet" if "scene0" in scene_name else "bundlefusion"][0].format(frame_index))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = ScannetDataset.fileformat_map["scannet" if "scene0" in folder else "bundlefusion"][1].format(frame_index)
        image_path = os.path.join(
            self.data_path,
            folder,
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = ScannetDataset.fileformat_map["scannet" if "scene0" in folder else "bundlefusion"][0].format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
