[![quick-segment](graphics/banner.svg)](https://gitlab.com/educelab/quick-segment)

## Installation

```shell
cd /path/to/quick-segment/
python3 -m venv venv
source venv/bin/activate

python -m pip install -e .
```

## Usage

```shell
source venv/bin/activate  # if you haven't already
quick-segment --input-volpkg <volpkg_path> --volume <volume_id>
```

## Updating the resources file
Use `rcc` provided by Qt6 to process `resources.qrc`. By default, this produces 
a file which imports PySide6, so make sure to modify the import for PyQt6.

```shell
cd qs/
rcc -g python resources.qrc | sed 's/PySide6/PyQt6/g' > resources.py
```

### macOS
For some reason, `rcc` isn't linked into the PATH. It can currently be found at:

```
/opt/homebrew/Cellar/qt/6.4.0_1/share/qt/libexec/rcc
```
