# Package Name
Add package description

### Installation
```bash
pip install package_name
```
Or
```bash
conda install -c edeno package_name
```
Or
```bash
git clone <package.git>
python setup.py install
```

### Usage

### Developer Installation
1. Install miniconda (or anaconda) if it isn't already installed.
2. git clone <package.git>
2. Setup editiable package with dependencies
```bash
cd <package folder>
conda env create -f environment.yml
conda activate package_name
python setup.py develop
```
