# small transformer trained on LTI systems
run train_onestep_wh --out-file ckpt_onestep_lin --cuda-device cuda:1

# big transformer trained on WH systems
run train_onestep_wh --out-file ckpt_onestep_wh_large --seq-len 1024  --n-layer 12 --n-head 12 --n-embd 768 --batch-size 20 --cuda-device cuda:1

# encoder-decoder simulation transformer on LTI systems
run train_sim_lin --bias --log-wandb

# encoder-decoder simulation transformer pre-trained on LTI systems adapted on WH systems
python train_sim_wh --bias --init-from pretrained --in-file ckpt_encdec_lin_sim --fixed-lr --lr 1e-4 --warmup-iters 0 --max-iters 5_000_000 --log-wandb

# checking the effect of a distribution shift
run train_sim_lin_shift --log-wandb --cuda-device cuda:2

# checking the effect of a distribution shift
run train_sim_lin_shift_mix --log-wandb --cuda-device cuda:1 --out-file "ckpt_shift_mix"