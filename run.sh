run_tag="deepcam_prediction_run1"
data_dir_prefix="deepCam/data"
output_dir="${data_dir_prefix}/runs/${run_tag}"

python ./deepCam/train_hdf5_ddp.py \
       --wireup_method "nccl-openmpi" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --model_prefix "classifier" \
       --optimizer "LAMB" \
       --start_lr 1e-3 \
       --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
       --lr_warmup_steps 0 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --validation_frequency 200 \
       --training_visualization_frequency 200 \
       --validation_visualization_frequency 40 \
       --max_validation_steps 50 \
       --logging_frequency 0 \
       --save_frequency 400 \
       --max_epochs 200 \
       --amp_opt_level O1 \
       --local_batch_size 2 |& tee -a ${output_dir}/train.out \
       > out.txt
