source ~/anaconda3/etc/profile.d/conda.sh 2> /dev/null
conda activate base 2> /dev/null
cd ..

export DFL_PYTHON="python"
export DFL_WORKSPACE="workspace/"
export DFL_ROOT="./"
export DFL_SRC="./DeepFaceLab_pytorch"
export DFL_DATASRC="data-source/"
export DFL_DATATGT="data-target/"

if [ "$#" -lt 2 ]; then
	echo "Insufficient number of parameters."
fi

origin_dir="$1"
torch_dir="$2"

$DFL_PYTHON "$DFL_SRC/main.py" converter \
    --origin-dir "$DFL_WORKSPACE/$origin_dir" \
    --torch-dir "$DFL_WORKSPACE/$origin_dir" \
    --config-name SAEHD_default_options.dat