# ./ipsr --in /home/pzzhao/zhengyu.wen/data/DTU/scan40/undistort/sparse/0/points3D.ply \
#     --out mesh_dtu40_30.ply \
#     --samplesPerNode 30.0 \
#     --depth 12

# ./ipsr/ipsr --in data/DTU/scan40/dense/0/fused.ply \
#     --out mesh_dtu40_dense_1.0.ply \
#     --samplesPerNode 1.0 \
#     --iters 100

cd ../..

./ipsr/ipsr --in mesh_dtu40_dense_1.4.ply \
    --out mesh_dtu40_dense_1.4_re.ply \
    --samplesPerNode 1.0 \
    --iters 100
