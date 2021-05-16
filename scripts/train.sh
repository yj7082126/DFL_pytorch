source ~/anaconda3/etc/profile.d/conda.sh 2> /dev/null
conda activate base 2> /dev/null
cd ..

export DFL_PYTHON="python"
export DFL_WORKSPACE="workspace/"
export DFL_ROOT="./"
export DFL_SRC="./DeepFaceLab_pytorch"
export DFL_DATASRC="data-source/"
export DFL_DATATGT="data-target/"

if [ "$#" -lt 4 ]; then
	echo "Insufficient number of parameters."
	exit 1
fi

source="$1"
target="$2"
model="$3"
config="$4"
savedmodel="$5"

$DFL_PYTHON "$DFL_SRC/main.py" train \
    --src-dir "$DFL_DATASRC/$source" \
    --dst-dir "$DFL_DATATGT/$target" \
    --model-dir "$DFL_WORKSPACE/$model" \
    --config-dir "$DFL_WORKSPACE/saved_configs/$config" \
    --device "0" \
    --savedmodel-dir "$DFL_WORKSPACE/saved_models/$savedmodel"
