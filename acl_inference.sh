CUDA_VISIBLE_DEVICES=5 python inference.py -loss_func acl -backbone resnet18 
CUDA_VISIBLE_DEVICES=5 python inference.py -loss_func acl -backbone dense
CUDA_VISIBLE_DEVICES=5 python inference.py -loss_func acl -backbone convnext_tiny