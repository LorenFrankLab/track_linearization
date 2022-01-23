[![PR Test](https://github.com/LorenFrankLab/track_linearization/actions/workflows/PR-test.yml/badge.svg)](https://github.com/LorenFrankLab/track_linearization/actions/workflows/PR-test.yml)

# track_linearization
Linearize 2D position to 1D using an HMM

### Installation
```bash
pip install track_linearization
```
Or
```bash
conda install -c franklab track_linearization
```
Or
```bash
git clone https://github.com/LorenFrankLab/track_linearization.git
python setup.py install
```

### Usage

### Developer Installation
1. Install miniconda (or anaconda) if it isn't already installed.
2. git clone https://github.com/LorenFrankLab/track_linearization.git
2. Setup editiable package with dependencies
```bash
cd <package folder>
conda env create -f environment.yml
conda activate track_linearization
python setup.py develop
```
