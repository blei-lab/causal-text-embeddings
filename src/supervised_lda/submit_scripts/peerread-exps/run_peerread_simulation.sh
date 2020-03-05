#!/usr/bin/env bash
#SBATCH -A sml
#SBATCH -c 8
#SBATCH --mail-user=dhanya.sridhar@columbia.edu
#SBATCH --mail-type=ALL

source activate py3.6

python -m supervised_lda.peerread_output_att \
--dat-dir=${DIR} \
--mode=${MODE} \
--params=${BETA1} \
--sim-dir=${SIMDIR} \
--outdir=${OUT}/${BETA1} \
--split=${SPLIT} \
--linear-outcome-model=${LINOUTCOME} \
--use-recon-loss=${RECONLOSS} \
--use-supervised-loss=${SUPLOSS} \