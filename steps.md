# Conda

conda create --name atari python=3.10

conda activate atari

# pipx

python -m pip install --user pipx
python -m pipx ensurepath

# poetry

pipx install poetry

# Git

git clone git@github.com:SuperSayf/cleanrl.git

cd cleanrl

poetry install

poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113

poetry install -E "atari jax"

poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

poetry run python cleanrl/dqn_atari_jax.py --env-id BreakoutNoFrameskip-v4
