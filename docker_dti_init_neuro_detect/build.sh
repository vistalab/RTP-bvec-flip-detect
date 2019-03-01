#!/bin/bash
# Builds the gear/container
# The container can be exported using the export.sh script
cd source/
. ./compile.sh
cd ../
git add .
git commit -m "Commiting before building $GEAR:$1"
GEAR=vistalab/neuro-detect
docker build --tag $GEAR:$1 .
