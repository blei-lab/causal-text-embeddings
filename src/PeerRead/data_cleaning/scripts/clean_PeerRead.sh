#!/bin/bash

# Process all PeerRead data into tf_record format to feed into Bert

PeerDir=../dat/PeerRead/

for dataset in $PeerDir*/; do
    echo $dataset
#    python -m data_cleaning.process_PeerRead_abstracts \
#    --review-json-dir \
#    --parsedpdf-json-dir \
#    --out-dir \
#    --out-file \
#    --vocab_file \
#    --max_abs_len
done