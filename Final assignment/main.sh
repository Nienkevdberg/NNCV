wandb login

python3 efficiency_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 20 \
    --lr 0.000075 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "E-train" \
    --teacher-checkpoint checkpoints/E-model.pt
