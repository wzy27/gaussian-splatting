export DATASET=$1 OUTPUT=$2 SCENE=$3 CUDA_VISIBLE_DEVICES=$4

python render.py -s data/$DATASET/$SCENE \
    -m $OUTPUT/$DATASET/$SCENE \
    --iteration 30000 --skip_train -w

# python metrics.py \
#     -m output/blender-all/$SCENE