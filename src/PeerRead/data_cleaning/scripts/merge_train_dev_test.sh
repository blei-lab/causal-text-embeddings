#!/bin/bash

# Each Peer read dataset is pre-divided into train/dev/test. Merge these into "all"

#PeerDir=../dat/PeerRead
PeerDir=../dat/PeerRead/nips_2013-2017

for dir in $PeerDir*/; do
    for subdir in $dir*/; do
	echo $subdir;
	cp -RT $subdir/ $dir/all/
    done
done
