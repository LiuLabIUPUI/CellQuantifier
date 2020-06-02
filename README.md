## Development installation

Create a new conda environment

```
conda create -n cq
conda activate cq
```

Install pip and its dependencies

```
conda install pip
```

Change to cellquantifier directory. This MUST be the parent directory
containing setup.py, not the cellquantifier subdirectory

```
cd /path/to/cellquantifier/
```

Install cellquantifier from source in dev mode

```
pip install -e .
```
