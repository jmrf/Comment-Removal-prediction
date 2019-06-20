#!/bin/bash

DIR="./workdir"

# Download model weights
wget \
    --max-redirect=20 \
    -O $DIR/download.zip \
    https://www.dropbox.com/sh/jgzggpwelcyicas/AAB5gTJ8Tn6T-mdpBNLZGtZPa?dl=0

# Unzip file
unzip $DIR/download.zip \
    -d $DIR

# remove the file
rm $DIR/download.zip
