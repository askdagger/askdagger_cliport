#!/bin/sh

#####################################
# TODO: Specify SBATCH requirements #
#####################################

ASKDAGGER=True

cd ${ASKDAGGER_ROOT}
source .venv/bin/activate

task_list="packing-seen-google-objects-seq packing-seen-google-objects-group packing-seen-shapes put-block-in-bowl-seen-colors"
task_i=$(($SLURM_ARRAY_TASK_ID / 60 + 1))
task=$(echo $task_list | cut -d ' ' -f $task_i)

srun python ${ASKDAGGER_ROOT}/src/askdagger_cliport/eval.py \
    iteration=$((($SLURM_ARRAY_TASK_ID / 6) % 10)) \
    eval.pier=$ASKDAGGER \
    relabeling_demos=$ASKDAGGER validation_demos=$ASKDAGGER \
    eval_task=$task model_task=$task \
    interactive_demos=$((50 + 50*($SLURM_ARRAY_TASK_ID % 6)))