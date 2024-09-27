#!/bin/sh
export PYTHONPATH=scripts:$PYTHONPATH

CUDA_VISIBLE_DEVICES='0,1' \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=29511 \
scripts/inference.py \
--step_inference 400 \
--sample_dir 'runs/cable' \
--large_recep 'weight/defect_gen/cable_large.pt' \
--small_recep 'weight/defect_gen/cable_small.pt' \
--num_defect 7 \
--large_recep_config 'config/large_recep_cabel.yml' \
--small_recep_config 'config/small_recep_cabel.yml' \
