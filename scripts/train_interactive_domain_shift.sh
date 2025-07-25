#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

PIER=True
FIER=True
task="packing-seen-shapes" 

srun python ${ASKDAGGER_ROOT}/src/askdagger_cliport/train_interactive_domain_shift.py \
    iteration=$SLURM_ARRAY_TASK_ID \
    train_interactive.pier=${PIER} \
    relabeling_demos=${FIER} validation_demos=${FIER} \
    train_interactive_task=$task model_task=$task \
    exp_folder=exps_domain_shift \
    interactive_demos=450
