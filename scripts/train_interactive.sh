#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

ASKDAGGER=True

cd ${ASKDAGGER_ROOT}
source .venv/bin/activate

task_list="packing-seen-google-objects-seq packing-seen-google-objects-group packing-seen-shapes put-block-in-bowl-seen-colors" 
task_i=$(($SLURM_ARRAY_TASK_ID / 10 + 1))
task=$(echo $task_list | cut -d ' ' -f $task_i)

srun python ${ASKDAGGER_ROOT}/src/askdagger_cliport/train_interactive.py \
    iteration=$(($SLURM_ARRAY_TASK_ID % 10)) \
    train_interactive.pier=$ASKDAGGER \
    relabeling_demos=$ASKDAGGER validation_demos=$ASKDAGGER \
    train_interactive_task=$task model_task=$task