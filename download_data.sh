#!/bin/bash

# downloads pre-processed ENCODE data
# tissue data from mouse and human
# histone modifications, CTCF, and DNase-seq + ATAC seq
# pre-processed metadata for sample groupings

mkdir -p data/processed

if ! command -v wget &> /dev/null
then
    echo "could not find wget, trying to use curl"

    download () {
        curl -L -o $1 $2
    }
    

else
    echo "using wget"

    download () {
        wget -O $1 $2
    }
    
fi

download  data/processed/encode_data.tar.gz https://figshare.com/ndownloader/files/38200983?private_link=60214b7914d6be3f5cb6 && \
cd data/processed && \
tar -xvzf encode_data.tar.gz
