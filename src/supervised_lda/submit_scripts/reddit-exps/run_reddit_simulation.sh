#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL

source activate py3.6

python -m supervised_lda.reddit_output_att \
--dat-dir=${DIR} \
--mode=${MODE} \
--subs=${SUBS} \
--params=${BETA0},${BETA1},${GAMMA} \
--sim-dir=${SIMDIR} \
--outdir=${OUT}/beta0${BETA0}.beta1${BETA1}.gamma${GAMMA} \
--split=${SPLIT} \
--linear-outcome-model=${LINOUTCOME} \
--use-recon-loss=${RECONLOSS} \
--use-supervised-loss=${SUPLOSS} \

