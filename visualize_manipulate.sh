#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --account=PAS1957
#SBATCH --gpus-per-node=1
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=28
export USER_HOME_DIR=~


if test -z $SLURM_JOB_ID 
then
    echo "then `date +%s`"
    export SLURM_JOB_ID=`date +%s`
fi

#export USER_JOB_NAME='22-06-19-204020'
if test -z $USER_JOB_NAME
then
    export USER_JOB_NAME=`date +%y-%m-%d-%H%M%S`
fi

echo $SLURM_JOB_ID
echo $USER_JOB_NAME
if test -z $OUTPUT_DIR
then
    export OUTPUT_DIR=output/$USER_JOB_NAME
fi


module load miniconda3
module load cuda

source activate
conda activate nlp
conda env list

set -x
#mkdir -p $OUTPUT_DIR
#mkdir -p $OUTPUT_DIR/code/
#cp *.py $OUTPUT_DIR/code/
#cp $0 $OUTPUT_DIR/code/

#python train_wavegan.py --model-size 64 --phase-shuffle-shift-factor 2 --post-proc-filt-len 512 --lrelu-alpha 0.2 --valid-ratio 0.1 --test-ratio 0.1 --batch-size 64 --num-epochs 3000 --batches-per-epoch 100 --ngpus 1 --latent-dim 100 --epochs-per-sample 1 --sample-size 20 --learning-rate 1e-4 --beta-one 0.5 --beta-two 0.9 --regularization-factor 10.0 --audio_dir=setTwo --output_dir=output --discriminator-updates=5 --job_id=$SLURM_JOB_ID > "sbatch/${SLURM_JOB_ID}_main.log"

# Manipulate latent variables verses nasal
python visualize_manipulate.py --output_dir=/users/PAS2062/delijingyic/project/wavegan/wavegan/output/22-07-07-111000 --input_dir=$USER_HOME_DIR/project/wavegan/wavegan/output/22-07-07-111000 #> "output/${SLURM_JOB_ID}/stdout.log"

# Manipulate latent variables verses s
# python manipulate.py --model-size 64 --post-proc-filt-len 512 --ngpus 1 --latent-dim 100 --output_dir=output --num_categ=3 --job_id=$SLURM_JOB_ID --model_path=/users/PAS2062/delijingyic/project/wavegan/output/17826262/model/649 --random_range=1 --num_epochs=2 --alter_axis=8,83,5,33,82,12,20 --alter_range=-15,-8,1 --filter_range=0.35,0.65 > "output/${SLURM_JOB_ID}/stdout.lo

mv output/$SLURM_JOB_ID.log $OUTPUT_DIR/sbatch.log