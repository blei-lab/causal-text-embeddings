#!/usr/bin/env bash

PeerDir=../dat/PeerRead/nips_2013-2017
PARSE_DIR=$PeerDir/2017/all/parsed_pdfs

for pdf in $PARSE_DIR/*; do
#    echo $pdf
    mv $pdf $PARSE_DIR/"${pdf#*/pdfs}"
done
