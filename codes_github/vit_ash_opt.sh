
DATASET_NAME="/home/damia001/vit/codes/train"
TRAIN_DIR="/home/damia001/vit/codes/train"
#OUTPUT_DIR="/home/damia001/vit/codes/ViT-augmented"
OUTPUT_DIR="/home/damia001/vit/codes/train_ImageFolder"

CUDA_DIVISIBLE_DEVICES=0, python3 run_image_classification.py \
    --train_dir $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    --do_train \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --overwrite_output_dir
