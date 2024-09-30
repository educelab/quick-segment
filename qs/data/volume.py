from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorstore as ts
from PIL import Image
from tqdm import tqdm


def _open_zarr(path, shape, chunk_size, create=False, delete_existing=False):
    extra = {}
    if create:
        extra['create'] = True
    if delete_existing:
        extra['delete_existing'] = True
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(path),
            },
            "metadata": {
                "shape": shape,
                "chunks": [chunk_size, chunk_size, chunk_size],
                "dtype": "<u2",
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": 10000000000,
                }
            }
        } | extra
    ).result()


class Volume:
    """
    NEW VOLUME LOADING AND MANAGING CLASS
    (Zarr or Slice Directory)
    """
    initialized_volumes: dict[str, Volume] = dict()

    @classmethod
    def from_path(cls, path: str, **kwargs) -> Volume:
        if path in cls.initialized_volumes.keys():
            return cls.initialized_volumes[path]
        cls.initialized_volumes[path] = Volume(path, **kwargs)
        return cls.initialized_volumes[path]

    def __init__(self, vol_path: str, load_zarr=True, save_zarr=False,
                 **kwargs):
        vol_path = Path(vol_path)
        self.path = vol_path

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

        zarr_path = vol_path / 'vol.zarr'
        if load_zarr and zarr_path.exists():
            logging.info(f'Using discovered vol.zarr')
            vol_path = vol_path / 'vol.zarr'

        chunk_size = 256
        if vol_path.suffix == ".zarr":
            self._is_zarr = True
            self._data = _open_zarr(vol_path,
                                    [self.shape_z, self.shape_y, self.shape_x],
                                    chunk_size)
        else:
            self._is_zarr = save_zarr
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

            futures = []
            if save_zarr:
                data = _open_zarr(zarr_path,
                                  [self.shape_z, self.shape_y, self.shape_x],
                                  chunk_size, create=True, delete_existing=True)

                def save_slice(i, image):
                    futures.append(data[i, :, :].write(image))
            else:
                data = np.empty((self.shape_z, self.shape_y, self.shape_x),
                                dtype=np.uint16)

                def save_slice(i, image):
                    data[i, :, :] = image

            for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
                img = np.array(Image.open(slice_file), dtype=np.uint16).copy()
                save_slice(slice_i, img)
                while len(futures) > 16:
                    for idx, f in enumerate(futures):
                        if f.done():
                            futures.pop(idx)

            # wait for futures if we need to
            if len(futures) > 0:
                logging.info("Waiting for .zarr to finish writing...")
            for f in futures:
                f.result()

            self._data = data

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

    def save_zarr(self, path: Union[str | Path | None] = None):
        if path is None:
            if self.path.suffix == ".zarr":
                logging.warning('Zarr already exists')
                return
            path = self.path / f'vol.zarr'
        elif isinstance(path, str):
            path = Path(path)

        if path.exists():
            logging.warning('Zarr already exists')
            return

        logging.info(f'Saving zarr volume: {str(path)}')
        chunk_size = 256
        data = ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(path),
                },
                "metadata": {
                    "shape": self.shape,
                    "chunks": [chunk_size, chunk_size, chunk_size],
                    "dtype": "<u2",
                },
                'create': True,
                'delete_existing': True
            }
        ).result()

        data[:].write(self._data[:]).result()
