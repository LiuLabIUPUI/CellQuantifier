## Development installation

Before installing, ensure that python=3.6.5 for full functionality.
This package depends on tensorflow-gpu=1.15 (compatible with CUDA 10.2) which
requires python=3.6.5. If conda is already installed, follow these instructions:

Create a new conda environment

```
conda create --name cq python=3.6.5
conda activate cq
```
Ensure pip is installed

```
conda install pip
```

Change to cellquantifier directory. This MUST be the parent directory
containing setup.py

```
cd /path/to/cellquantifier/
```

Install cellquantifier from source in dev mode

```
pip install -e .
```
