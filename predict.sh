#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --account=PAS1957
#SBATCH --gpus-per-node=1
#SBATCH --output=output/%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=28


if test -z $SLURM_JOB_ID 
then
    echo "then `date +%s`"
    export SLURM_JOB_ID=`date +%s`
fi

echo $SLURM_JOB_ID
export USER_HOME_DIR=~


module load miniconda3
module load cuda/10.2.89

source activate
conda activate nlp
conda env list

set -x
mkdir output/$SLURM_JOB_ID
mkdir output/$SLURM_JOB_ID/code/
cp *.py output/$SLURM_JOB_ID/code/
cp $0 output/$SLURM_JOB_ID/code/

#python train_wavegan.py --model-size 64 --phase-shuffle-shift-factor 2 --post-proc-filt-len 512 --lrelu-alpha 0.2 --valid-ratio 0.1 --test-ratio 0.1 --batch-size 64 --num-epochs 3000 --batches-per-epoch 100 --ngpus 1 --latent-dim 100 --epochs-per-sample 1 --sample-size 20 --learning-rate 1e-4 --beta-one 0.5 --beta-two 0.9 --regularization-factor 10.0 --audio_dir=setTwo --output_dir=output --discriminator-updates=5 --job_id=$SLURM_JOB_ID > "sbatch/${SLURM_JOB_ID}_main.log"
#python predict.py --model-size 64 --post-proc-filt-len 512 --batch-size 64 --ngpus 1 --latent-dim 100 --output_dir=output --num_categ=3 --job_id=$SLURM_JOB_ID --model_path=/users/PAS2062/delijingyic/project/org_wavegan/output/17826262/model/649 --random_range=1 --num_epochs=60 > "output/${SLURM_JOB_ID}/stdout.log"

python predict.py --model-size 64 --post-proc-filt-len 512 --batch-size 64 --ngpus 1 --latent-dim 100 --output_dir=output --num_categ=3 --job_id=$SLURM_JOB_ID --model_path=/users/PAS2062/delijingyic/project/wavegan/wavegan/output/20535258/model/977 --random_range=1 --num_epochs=60 > "output/${SLURM_JOB_ID}/stdout.log"

mv output/$SLURM_JOB_ID.log output/$SLURM_JOB_ID/sbatch.log



