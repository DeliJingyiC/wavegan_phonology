#!/bin/bash

#SBATCH --time=1:00:00
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


module load miniconda3/4.10.3-py37
module load cuda/10.2.89

source activate
conda activate tsf114
conda env list

set -x
mkdir output/$SLURM_JOB_ID
mkdir output/$SLURM_JOB_ID/code/
cp *.py output/$SLURM_JOB_ID/code/
cp $0 output/$SLURM_JOB_ID/code/


python linear_regression.py --latent_v_dir=/users/PAS2062/delijingyic/project/wavegan/wavegan/output/19346227/latent_v --s_dir=/users/PAS2062/delijingyic/project/wavegan/nasalDNN/output/20691436/VN_output/VN_output.csv --job_id=$SLURM_JOB_ID --output_dir=output > "output/${SLURM_JOB_ID}/stdout.log"

mv output/$SLURM_JOB_ID.log output/$SLURM_JOB_ID/sbatch.log
