#!/bin/bash

set -e

# Install DPDP AE-RNN
if [ ! -d ../dpdp_aernn ]; then
    git clone https://github.com/kamperh/dpdp_aernn ../dpdp_aernn
fi

# Install VectorQuantizedCPC
if [ ! -d ../VectorQuantizedCPC ]; then
    git clone https://github.com/kamperh/VectorQuantizedCPC \
        ../VectorQuantizedCPC
fi

# Install VectorQuantizedVAE
if [ ! -d ../VectorQuantizedVAE ]; then
    git clone https://github.com/kamperh/ZeroSpeech ../VectorQuantizedVAE
fi
