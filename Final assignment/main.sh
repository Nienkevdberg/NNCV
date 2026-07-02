wandb login

python3 PP_train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 20 \
    --lr 0.000075 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "PP-train" \
