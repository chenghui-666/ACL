CUDA_VISIBLE_DEVICES=1 python train.py -loss_func acl -backbone resnet18 -HQ_len 16 -coff_std 0
CUDA_VISIBLE_DEVICES=1 python train.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 0
CUDA_VISIBLE_DEVICES=1 python train.py -loss_func acl -backbone resnet18 -HQ_len 64 -coff_std 0
CUDA_VISIBLE_DEVICES=1 python train.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 1
CUDA_VISIBLE_DEVICES=1 python train.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 2

# CUDA_VISIBLE_DEVICES=1 python inference.py -loss_func acl -backbone resnet18 -HQ_len 16 -coff_std 0
# CUDA_VISIBLE_DEVICES=1 python inference.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 0
# CUDA_VISIBLE_DEVICES=1 python inference.py -loss_func acl -backbone resnet18 -HQ_len 64 -coff_std 0
# CUDA_VISIBLE_DEVICES=1 python inference.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 1
# CUDA_VISIBLE_DEVICES=1 python inference.py -loss_func acl -backbone resnet18 -HQ_len 32 -coff_std 2