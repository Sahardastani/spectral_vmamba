
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_addr="127.0.0.1" \
    --master_port=21495 \
    main.py \
    --cfg configs/vssm/vmamba_tiny_224.yaml \
    --batch-size 128 \
    --data-path /data/shared/mini-imagenet \
    --output output