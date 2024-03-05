export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-octree.py --eval -s \
    /data/nglm005/zhengyu.wen/data/Omni/sofa010 \
    --sh_degree 0 \
    --iterations 15000 \
    --epochs 10 \
    --opacity_reset_interval 300 \
    --lambda_orientation 1 \
    --lambda_scale 1000

    # --hessian_eikonal