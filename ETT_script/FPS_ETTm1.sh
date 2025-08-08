export CUDA_VISIBLE_DEVICES=0

model_name=FPS #DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 256 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1