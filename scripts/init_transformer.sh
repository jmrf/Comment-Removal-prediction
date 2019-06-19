#!/bin/bash

DIR="./external/models/transformer"

# Download model weights
wget \
    --max-redirect=20 \
    -O $DIR.zip \
    https://www.dropbox.com/sh/m32vct5txebd9kv/AABRbIhdCZOyjfDthqhTjCXTa?dl=0

# Unzip file
unzip $DIR.zip \
    -d $DIR

# remove the file
rm $DIR.zip
