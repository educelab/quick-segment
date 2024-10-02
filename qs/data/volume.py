from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union
from os import PathLike

import numpy as np
import tensorstore as ts
from PIL import Image
from tqdm import tqdm
import psutil


def _create_zarr(path: Union[str, PathLike],
                 shape: Union[Tuple[int, int, int], List[int]],
                 chunk_shape: Union[Tuple[int, int, int], List[int]],
                 delete_existing: bool = False,
                 cache_bytes: int = None):
    extra = {}
    if delete_existing:
        extra['delete_existing'] = True
    if cache_bytes is None:
        cache_bytes = 1_000_000_000
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(path),
            },
            "metadata": {
                "shape": shape,
                "chunks": chunk_shape,
                "dtype": "<u2",
                "fill_value": 0,
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": cache_bytes,
                }
            },
            "create": True,
            "recheck_cached_data": "open"
        } | extra
    ).result()


def _load_zarr(path: Union[str, PathLike], cache_bytes: int = None):
    if cache_bytes is None:
        cache_bytes = 1_000_000_000
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(path),
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": cache_bytes,
                }
            },
            "recheck_cached_data": "open"
        }
    ).result()


class Volume:
    """
    NEW VOLUME LOADING AND MANAGING CLASS
    (Zarr or Slice Directory)
    """
    initialized_volumes: dict[str, Volume] = dict()

    @classmethod
    def from_path(cls, path: Union[str, PathLike], **kwargs) -> Volume:
        spath = str(path)
        if spath in cls.initialized_volumes.keys():
            return cls.initialized_volumes[spath]
        cls.initialized_volumes[spath] = Volume(path, **kwargs)
        return cls.initialized_volumes[spath]

    def __init__(self, vol_path: Union[str, PathLike], load_zarr=True,
                 save_zarr=False,
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
        data_shape = (self.shape_z, self.shape_y, self.shape_x)

        zarr_path = vol_path / 'vol.zarr'
        if load_zarr and zarr_path.exists():
            logging.info(f'Using discovered vol.zarr')
            vol_path = vol_path / 'vol.zarr'

        # loader cache size max(1x slice size, 5% of RAM)
        # TODO: Parameterize?
        slice_bytes = np.prod(data_shape[1:]) * 2
        max_bytes = max(slice_bytes, psutil.virtual_memory().total // 20)

        # Create chunk size
        # TODO: Need a heuristic for this. See h5py?
        chunk_size = [8, 256, 256]

        # Load the existing zarr
        if vol_path.suffix == ".zarr":
            self._is_zarr = True
            self._data = _load_zarr(vol_path,
                                    cache_bytes=zarr_cache_bytes)
        # Load the raw slices
        else:
            # Use the zarr after load if we're saving it
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

            # Set up our slice save function depending on whether we're loading
            # into memory or a new zarr
            if save_zarr:
                # How many slices we can load w/o exceeding the loader cache
                slice_batch_size = max_bytes // slice_bytes

                # If our slice batch size is bigger than the Z chunk dim,
                # shrink the batch size to a multiple of that dim to avoid
                # writing (and rewriting) partial chunks to disk
                if slice_batch_size > chunk_size[0]:
                    slice_batch_size = chunk_size[0] * (
                            slice_batch_size // chunk_size[0])

                # Create the zarr
                data = _create_zarr(zarr_path, data_shape, chunk_size,
                                    delete_existing=True,
                                    cache_bytes=zarr_cache_bytes)

                # Save slice range function for zarr
                def save_slice(start, end, image):
                    data[start:end, :, :].write(image).result()
            else:
                # Always one slice at a time
                slice_batch_size = 1
                # Create the in-memory volume
                data = np.empty((self.shape_z, self.shape_y, self.shape_x),
                                dtype=np.uint16)

                # Save slice range function for np.ndarray
                def save_slice(start, end, image):
                    data[start:end, :, :] = image

            # Load slice images into volume
            logging.info(f"Loading volume slices from {vol_path}...")
            batch = np.empty((min(self.shape_z, slice_batch_size), self.shape_y,
                              self.shape_x), dtype=np.uint16)
            batch_start = 0
            for slice_i, slice_file in tqdm(list(enumerate(slice_files))):
                # Load the images
                img = np.array(Image.open(slice_file))
                # Fallback to single slice saving if batch size is too small
                if slice_batch_size <= 1:
                    save_slice(batch_start, batch_start + 1, img)
                    batch_start += 1
                # Add to the slice batch
                else:
                    batch[slice_i - batch_start] = img
                    # If our batch is full, write to the output
                    if (slice_i - batch_start) == slice_batch_size - 1:
                        batch_end = batch_start + slice_batch_size
                        save_slice(batch_start, batch_end, batch)
                        batch_start = batch_end
                        # next batch size if the min of remaining slices and current batch size
                        slice_batch_size = min(self.shape_z - batch_start,
                                               slice_batch_size)
                        del batch
                        if slice_batch_size > 1:
                            batch = np.empty(
                                (slice_batch_size, self.shape_y, self.shape_x),
                                dtype=np.uint16)

            # Store the handle to the data array
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
