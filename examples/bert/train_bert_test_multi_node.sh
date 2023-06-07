[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
run_name=bert_example
save_dir="./save/${run_name}"
mkdir -p ${save_dir}

torchrun --nproc_per_node=$MLP_WORKER_GPU --nnodes=$MLP_WORKER_NUM  --node_rank=$MLP_ROLE_INDEX  --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
        $(which unicore-train) ./example_data  --user-dir . --valid-subset valid \
       --num-workers 0 --ddp-backend=c10d \
       --task bert --loss masked_lm --arch bert_base  \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr 1e-4 --warmup-updates 100 --total-num-update 10000 --batch-size 4 \
       --update-freq 1 --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update 10000 --log-interval 100 --log-format simple \
       --save-interval-updates 1000 --validate-interval-updates 1000 --keep-interval-updates 30 --no-epoch-checkpoints  \
       --save-dir $save_dir


