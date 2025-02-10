python -m torch.distributed.launch --nproc_per_node=1 --master_port=4327 basicsr/test.py \
        -opt options/test/RB2V/NAFNet-width64.yml \
        --launcher pytorch