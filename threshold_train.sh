export DATASET_DIR=$1 OUTPUT_DIR=$2 SCENE=$3 CUDA_VISIBLE_DEVICES=$4
export THRE=$5 SCALAR=$6
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-octree.py \
    --eval -w \
    -s data/$DATASET_DIR/$SCENE \
    -m $OUTPUT_DIR/$DATASET_DIR/$SCENE \
    --iterations 30000 \
    --lambda_opacity 3 \
    --lambda_orientation 0 \
    --lambda_scale 0.1 \
    --near_threshold $THRE \
    --opacity_scalar $SCALAR

    # --hessian_eikonal