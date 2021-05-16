source ~/anaconda3/etc/profile.d/conda.sh 2> /dev/null
conda activate base 2> /dev/null
cd ..

export DFL_PYTHON="python"
export DFL_WORKSPACE="workspace"
export DFL_ROOT="./"
export DFL_SRC="./DeepFaceLab_pytorch"
export DFL_DATASRC="data-source"
export DFL_DATATGT="data-target"
export DFL_VIDINP="video-input"
export DFL_VIDOUT="video-output"

if [ "$#" -lt 1 ]; then
	echo "Insufficient number of parameters."
	exit 1
fi

video="$1"
videoname="$(cut -d'.' -f1 <<< "$video")"
type="$2"

$DFL_PYTHON "$DFL_SRC/main.py" videoed extract-video \
    --input-file "$DFL_VIDINP/$video" \
    --output-dir "$DFL_VIDINP/$videoname" \
    --output-ext png

$DFL_PYTHON "$DFL_SRC/main.py" extract \
    --input-dir "$DFL_VIDINP/$videoname" \
    --output-dir "$DFL_VIDINP/$videoname/aligned" \
    --input-ext png

if [ $type = "source" ]; then
    mkdir -p "$DFL_DATASRC/$videoname"
	mv "$DFL_VIDINP/$videoname/aligned/" "$DFL_DATASRC/$videoname"
else
    mkdir -p "$DFL_DATATGT/$videoname"
	cp -r "$DFL_VIDINP/$videoname/aligned/" "$DFL_DATATGT/$videoname"
fi 

