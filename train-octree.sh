export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-octree.py --eval -w -s \
    /data/nglm005/zhengyu.wen/data/blender/chair \
    --sh_degree 0 \
    --iterations 15000 \
    --epochs 10 \
    --opacity_reset_interval 300 \
    --lambda_orientation 0 \
    --lambda_scale 0\
    --lambda_opacity 1\
    --opacity_reset_interval 300\
    --checkpoint_iterations 15000\
    # --hessian_eikonal