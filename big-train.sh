export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PYTHONPATH:/home/nglm005/data/nglm005/zhengyu.wen/LOTree

python train-big.py --eval -s \
    /home/nglm005/data/nglm005/zhengyu.wen/data/Omni/table024 \
    --sh_degree 0 \
    --iterations 3000 \
    --epochs 10


# tmux:29