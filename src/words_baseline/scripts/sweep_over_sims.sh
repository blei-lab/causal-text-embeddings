#!/usr/bin/env bash
#NUM_SEED=2
#SEEDS=$(seq 0 $NUM_SEED)
rm ../dat/reddit/sim/reddit_subreddit_based/two-stage-lda-estimates.out
export SUBREDDITS=13,6,8
export BETA0=1.0
declare -a SIMMODES=('simple')
declare -a BETA1S=(1.0 10.0 100.0)
declare -a GAMMAS=(1.0 4.0)

for SIMMODEj in "${SIMMODES[@]}"; do
    for BETA1j in "${BETA1S[@]}"; do
        for GAMMAj in "${GAMMAS[@]}"; do
            python -m lda_baseline.reddit_output_att \
            --subs=${SUBREDDITS} \
            --mode=${SIMMODEj} \
            --params=${BETA0},${BETA1j},${GAMMAj}
        done
    done
done