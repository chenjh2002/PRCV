PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --master_port 20000 --nproc_per_node 4 ./train.py --dataset cifar10 --num-labeled 400 \
            --arch wideresnet  --batch-size 16 --lr 0.03 --wdecay 0.001 \
            --expand-labels --seed 5 --out results/cifar10@400