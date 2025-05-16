torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    train_cross_encoder.py