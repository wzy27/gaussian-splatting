export DATASET_DIR=$1 OUTPUT_DIR=$2 SCENE=$3 CUDA_VISIBLE_DEVICES=$4
export OPACITY=$5 SCALE=$6
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-octree.py \
    --eval \
    -s data/$DATASET_DIR/$SCENE \
    -m $OUTPUT_DIR/$DATASET_DIR/$SCENE \
    --iterations 30000 \
    --lambda_opacity $OPACITY \
    --lambda_orientation 0 \
    --lambda_scale $SCALE \
    --opacity_scalar 10 \
    --near_threshold 0.05 \
    -w

    # --hessian_eikonal