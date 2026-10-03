export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

torchrun --nproc_per_node 2 --master_port 5115 /your/path/to/train_coffee_mate.py \
    /your/path/to/scripts/config_coffee_mate.py \
    output_dir /your/path/to/log/
