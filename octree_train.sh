export DATASET_DIR=$1 SCENE=$2 CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-octree.py \
    --eval \
    -s data/$DATASET_DIR/$SCENE \
    -m output-ours-SH/$DATASET_DIR/$SCENE \
    --iterations 30000 \
    --lambda_opacity 3 \
    --lambda_scale 0.1  \
    --opacity_scalar 200 \
    --near_threshold 0.01 \
    -w

    # --hessian_eikonal