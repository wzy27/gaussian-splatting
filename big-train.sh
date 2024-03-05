export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu

python train-big.py --eval -s \
    /data/nglm005/zhengyu.wen/data/Omni/dino006 \
    --sh_degree 0 \
    --iterations 3000 \
    --epochs 10 \
    --opacity_reset_interval 300 \
    --lambda_orientation 0 \
    --lambda_scale 0 \
    --gaussian_smooth \
    --densify_until_iter 30000

    # --hessian_eikonal


# tmux:29