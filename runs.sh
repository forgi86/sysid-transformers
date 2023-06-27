# big transformer trained on wh systems
run main_transformer_wh --out-file ckpt_big --seq-len 1024  --n-layer 12 --n-head 12 --n-embd 768 --batch-size 20 --cuda-device cuda:1 

# big transformer tuned to parallel wh model structure
run main_transformer_pwh --init-from pretrained --in-file ckpt_big --out-file ckpt_big_pwh --batch-size 20 --fixed-lr --warmup-iters 0 --cuda-device cuda:1

# small transformer tuned to a specific wh model
run main_transformer_wh --fixed-system --seed 1 --init-from pretrained --in-file ckpt_small_wh --out-file ckpt_small_wh_adapt --max-iter 201 --fixed-lr --lr 1e-4 --warmup-iters 0 --cuda-device cuda:0

# big transformer tuned to a specific wh model
run main_transformer_wh --fixed-system --seed 1 --init-from pretrained --in-file ckpt_big_wh --out-file ckpt_big_wh_adapt --max-iter 201 --batch-size 20 --fixed-lr --lr 1e-4 --warmup-iters 0 --cuda-device cuda:0

# small transformer learned on a specific wh model
run main_transformer_wh --fixed-system --seed 1 --init-from scratch --out-file ckpt_small_wh_single --max-iter 6000 --fixed-lr --lr 1e-4 --warmup-iters 0 --cuda-device cuda:0 --log-wandb

# encoder-decoder simulation transformer on linear systems
run main_transformer_lin_encdec_sim --bias --log-wandb