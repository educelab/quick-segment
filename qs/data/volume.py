from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorstore as ts
from PIL import Image
from tqdm import tqdm
import psutil


def _open_zarr(path, shape, chunk_size, create=False, delete_existing=False,
               cache_bytes=None):
    extra = {}
    if create:
        extra['create'] = True
    if delete_existing:
        extra['delete_existing'] = True
    if cache_bytes is None:
        cache_bytes = psutil.virtual_memory().total // 2
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
                    "total_bytes_limit": cache_bytes,
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
                 zarr_cache_bytes=None, **kwargs):
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
        data_shape = [self.shape_z, self.shape_y, self.shape_x]

        zarr_path = vol_path / 'vol.zarr'
        if load_zarr and zarr_path.exists():
            logging.info(f'Using discovered vol.zarr')
            vol_path = vol_path / 'vol.zarr'

        # loader cache size (4GB)
        max_bytes = 4000000000
        slice_bytes = np.prod(data_shape[1:]) * 2

        chunk_size = 256
        if vol_path.suffix == ".zarr":
            self._is_zarr = True
            self._data = _open_zarr(vol_path,
                                    data_shape,
                                    chunk_size,
                                    cache_bytes=zarr_cache_bytes)
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
            logging.info(f"Loading volume slices from {vol_path}...")

            if save_zarr:
                slice_batch_size = max_bytes // slice_bytes
                # if slice is bigger than loader mem limit, load a single slice
                if slice_batch_size == 0:
                    slice_batch_size = 1
                # shrink the batch size to a multiple of the chunk size to avoid file rewrites
                elif slice_batch_size > chunk_size:
                    slice_batch_size = chunk_size * (
                                slice_batch_size // chunk_size)
                data = _open_zarr(zarr_path,
                                  [self.shape_z, self.shape_y, self.shape_x],
                                  chunk_size, create=True, delete_existing=True,
                                  cache_bytes=zarr_cache_bytes)

                def save_slice(start, end, image):
                    data[start:end, :, :].write(image).result()
            else:
                slice_batch_size = 1
                data = np.empty((self.shape_z, self.shape_y, self.shape_x),
                                dtype=np.uint16)

                def save_slice(start, end, image):
                    data[start:end, :, :] = image
            logging.debug(f'Slice batch size: {slice_batch_size}')
            batch = []
            batch_start = 0
            for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
                img = np.array(Image.open(slice_file), dtype=np.uint16).copy()
                if slice_batch_size == 1:
                    save_slice(batch_start, batch_start + 1, img)
                else:
                    batch.append(img)
                    if len(batch) == slice_batch_size:
                        batch = np.stack(batch)
                        batch_end = batch_start + slice_batch_size
                        save_slice(batch_start, batch_end, batch)
                        batch_start = batch_end
                        batch = []

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
