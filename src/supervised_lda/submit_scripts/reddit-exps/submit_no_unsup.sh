#!/usr/bin/env bash
BASE_OUT=/proj/sml_netapp/projects/causal-text/reddit/supervised_lda_baseline/out/

export DIR=/proj/sml_netapp/projects/causal-text/reddit/supervised_lda_baseline/proc/
export SIMDIR=/proj/sml_netapp/projects/causal-text/sim/reddit_subreddit_based/

export MODE=simple
export SUBS=13,6,8
export LINOUTCOME=t
export RECONLOSS=f
export SUPLOSS=t

export BETA0=1.0
declare -a BETA1S=(1.0 10.0 100.0)
declare -a GAMMAS=(1.0 4.0)

for BETA1j in "${BETA1S[@]}"; do
	export BETA1=${BETA1j}
	for GAMMAj in "${GAMMAS[@]}"; do
		for SPLITi in $(seq 0 4); do
			export SPLIT=${SPLITi}
			export GAMMA=${GAMMAj}
			export OUT=${BASE_OUT}/no_unsup/
			sbatch --job-name=reddit_supervised_lda_sim_${BETA1j}_${GAMMAj}_${SPLITi} \
				   --output=reddit_supervised_lda_sim_${BETA1j}_${GAMMAj}_${SPLITi}.out \
				   supervised_lda/submit_scripts/reddit-exps/run_reddit_simulation.sh

	   done
	done
done
