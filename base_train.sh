export DATASET_DIR=$1 SCENE=$2 CUDA_VISIBLE_DEVICES=$3 OUTPUT_DIR=$4

BASEPORT=12235
PORT=$(($CUDA_VISIBLE_DEVICES + $BASEPORT))

python train-base.py -s data/$DATASET_DIR/$SCENE \
    -m $OUTPUT_DIR/$DATASET_DIR/$SCENE \
    --eval \
    --port $PORT \
    -w

# python metrics.py \
#     -m output/blender-all/$SCENE