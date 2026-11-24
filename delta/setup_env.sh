#!/bin/bash
# run once on a delta login node, from the repo root: bash delta/setup_env.sh
# versions are the ones the pipeline was tested with on oct 22-23.
set -euo pipefail

FORGE="$HOME/miniforge3"
if [ ! -x "$FORGE/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$FORGE"
  rm /tmp/miniforge.sh
fi
source "$FORGE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^cytox\s"; then
  conda create -y -n cytox -c conda-forge --override-channels python=3.13 pip
fi
conda activate cytox

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "scvi-tools==1.5.1" "decoupler==2.2.0" scanpy anndata matplotlib
pip install "git+https://github.com/theislab/scgen.git"

# compute nodes may not have internet, so cache the collectri network now
python -c "from collecTRI import load_collectri; print(load_collectri().shape)"
python -c "import scgen, decoupler, scvi; print('ok', scvi.__version__, decoupler.__version__)"
