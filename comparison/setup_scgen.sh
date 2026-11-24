#!/usr/bin/env bash
# Sets up the scgen-repro environment (Python 3.7 + TensorFlow 1.15) on an
# x86_64 Linux machine: GitHub Codespaces, NCSA Delta, or any Linux box.
#
# Usage (from the repo root):
#   bash setup_scgen.sh
#   source ~/miniforge3/etc/profile.d/conda.sh && conda activate scgen-repro
set -euo pipefail

ENV_NAME=scgen-repro
FORGE_DIR="$HOME/miniforge3"

# 1. Install Miniforge (conda-forge only, so no Anaconda ToS prompts)
if [ ! -x "$FORGE_DIR/bin/conda" ]; then
  echo ">> Installing Miniforge into $FORGE_DIR"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$FORGE_DIR"
  rm /tmp/miniforge.sh
fi
source "$FORGE_DIR/etc/profile.d/conda.sh"

# 2. Create the Python 3.7 environment
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  echo ">> Creating $ENV_NAME (Python 3.7)"
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.7 pip
fi
conda activate "$ENV_NAME"

# 3. Install pinned packages (this exact set was tested to resolve and import)
echo ">> Installing packages"
python -m pip install --upgrade "pip<24.1"
python -m pip install \
  "tensorflow>=1.15,<2" "keras==2.3.1" "protobuf<3.21" "h5py<3" \
  "numpy<1.19" "scipy>=1.3,<1.5" "pandas<1.0" \
  "scanpy==1.4.4.post1" "anndata==0.6.22.post1" \
  "get-version==2.1" adjustText \
  matplotlib seaborn scikit-learn wget jupyter ipykernel

# 4. Register a Jupyter kernel so the notebooks can use this env
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 5. Smoke tests
echo ">> Checking TensorFlow"
python -c "import tensorflow as tf; print('TF', tf.__version__, '->', tf.Session().run(tf.reduce_sum(tf.ones(3))))" 2>/dev/null
echo ">> Checking scgen import"
(cd code && python -c "import scgen, scanpy, anndata; print('scgen OK | scanpy', scanpy.__version__, '| anndata', anndata.__version__)" 2>/dev/null)

# 6. Record exact versions for your methods section
python -m pip freeze > environment-lock.txt
echo ">> Done. Exact versions saved to environment-lock.txt"
echo ">> Next: source $FORGE_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
