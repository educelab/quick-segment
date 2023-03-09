from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorstore as ts
from PIL import Image
from tqdm import tqdm


class Volume:
    """
    NEW VOLUME LOADING AND MANAGING CLASS
    (Zarr or Slice Directory)
    """
    initialized_volumes: dict[str, Volume] = dict()

    @classmethod
    def from_path(cls, path: str) -> Volume:
        if path in cls.initialized_volumes:
            return cls.initialized_volumes[path]
        cls.initialized_volumes[path] = Volume(path)
        return cls.initialized_volumes[path]

    def __init__(self, vol_path: str):
        vol_path = Path(vol_path)

        # Load metadata
        self._metadata = dict()
        metadata_filename = vol_path / "meta.json"
        if not metadata_filename.exists():
            raise FileNotFoundError(
                f"No volume meta.json file found in {vol_path}")
        else:
            with open(metadata_filename) as f:
                self._metadata = json.loads(f.read())
        self._voxelsize_um = self._metadata["voxelsize"]
        self.shape_z = self._metadata["slices"]
        self.shape_y = self._metadata["height"]
        self.shape_x = self._metadata["width"]

        if vol_path.suffix == ".zarr":
            self._is_zarr = True
            chunk_size = 256
            self._data = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": str(vol_path),
                    },
                    "metadata": {
                        "shape": [self.shape_z, self.shape_y, self.shape_x],
                        "chunks": [chunk_size, chunk_size, chunk_size],
                        "dtype": "<u2",
                    },
                    "context": {
                        "cache_pool": {
                            "total_bytes_limit": 10000000000,
                        }
                    }
                }
            ).result()
        else:
            self._is_zarr = False
            # Get list of slice image filenames
            slice_files = []
            for child in vol_path.iterdir():
                if not child.is_file():
                    continue
                # Make sure it is not a hidden file and it's a .tif
                if child.name[0] != "." and child.suffix == ".tif":
                    slice_files.append(str(child))
            slice_files.sort()
            assert len(slice_files) == self.shape_z

            # Load slice images into volume
            logging.info("Loading volume slices from {}...".format(vol_path))

            self._data = np.empty(
                (self.shape_z, self.shape_y, self.shape_x),
                dtype=np.uint16
            )
            for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
                img = np.array(Image.open(slice_file), dtype=np.uint16).copy()
                self._data[slice_i, :, :] = img
            print()

    def __getitem__(self, key):
        # TODO consider adding bounds checking and return 0 if not in bounds (to match previous implementation)
        #   It would be nice to avoid that if possible (doesn't affect ML performance), though, because
        #   it breaks the intuition around the array access.
       
        if self._is_zarr:
            return self._data[key].read().result()
        else:
            return self._data[key]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
