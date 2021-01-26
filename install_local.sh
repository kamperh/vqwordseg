#!/bin/bash

set -e

# Install VectorQuantizedCPC
if [ ! -d ../VectorQuantizedCPC ]; then
    git clone https://github.com/kamperh/VectorQuantizedCPC \
        ../VectorQuantizedCPC
fi

# Install VectorQuantizedVAE
if [ ! -d ../VectorQuantizedVAE ]; then
    git clone https://github.com/kamperh/ZeroSpeech ../VectorQuantizedVAE
fi
