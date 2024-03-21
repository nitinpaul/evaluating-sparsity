import os
import re

import cv2
import numpy as np
import pickle as pkl
import torch.utils.data

import transform

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


class DataSet(torch.utils.data.Dataset):
    """Provides dataset capabilities for image retrieval tasks."""

    def __init__(self, data_path, dataset, fn, split, scale_list):
        """
        Args:
            data_path: Path to the main data directory.
            dataset: Name of the dataset (e.g., 'oxford5k', 'roxford5k').
            fn: Filename containing ground truth data.
            split: Whether it's the 'query' or 'db' split.
            scale_list: List of scales for multi-scale image retrieval.
        """
        assert os.path.exists(data_path), f"Data path '{data_path}' not found"
        self._data_path = data_path
        self._dataset = dataset
        self._fn = fn
        self._split = split
        self._scale_list = scale_list
        self._construct_db()

    def _construct_db(self):
        """Constructs the internal image database."""
        self._db = []

        if self._dataset in ["oxford5k", "roxford5k", "paris6k", "rparis6k"]:
            split_path = os.path.join(self._data_path, self._dataset, self._fn)
            with open(split_path, "rb") as fin:
                gnd = pkl.load(fin)  # Load ground truth

            if self._split == "query":
                for i in range(len(gnd["qimlist"])):
                    image_path = os.path.join(
                        self._data_path, self._dataset, "jpg", gnd["qimlist"][i] + ".jpg"
                    )
                    self._db.append(
                        {"im_path": image_path, "bbox": gnd["gnd"][i]["bbx"]}
                    )

            elif self._split == "db":
                for i in range(len(gnd["imlist"])):
                    image_path = os.path.join(
                        self._data_path, self._dataset, "jpg", gnd["imlist"][i] + ".jpg"
                    )
                    self._db.append({"im_path": image_path})

        else:
            raise ValueError(f"Unsupported dataset: {self._dataset}")

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])  # Channels first (e.g., for PyTorch)
        im = im / 255.0  
        im = transform.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        """Retrieves an image with multi-scale versions."""
        try:
            image_info = self._db[index]
            im = cv2.imread(image_info["im_path"])

            if self._split == "query":
                bbox = image_info["bbox"]
                im = im[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]  

            im_list = []
            for scale in self._scale_list:
                if scale == 1.0:
                    im_np = im.astype(np.float32, copy=False)

                else:
                    # Choose interpolation based on upscaling/downscaling
                    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                    im_resize = cv2.resize(im, dsize=(0, 0), fx=scale, fy=scale, interpolation=interp)
                    im_np = im_resize.astype(np.float32, copy=False)

                im_list.append(im_np)

            # Prepare and return images at multiple scales
            return [self._prepare_im(im) for im in im_list]

        except Exception as e:  # Consider more specific exception handling
            print(f"Error loading image: {image_info['im_path']}")
            raise e  # Re-raise the exception for the Dataset to handle it

    def __len__(self):
        return len(self._db)
