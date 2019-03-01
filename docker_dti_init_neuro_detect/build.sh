#!/bin/bash
# Builds the gear/container
# The container can be exported using the export.sh script
echo "You introduced version $1"
read -p "Continue (y/n)?" choice
case "$choice" in 
    y|Y ) echo "yes";;
    n|N ) echo "no";;
    * ) echo "invalid";;
esac
cd source/
. ./compile.sh
cd ../
git add .
git commit -m "Commiting before building $GEAR-$1"
GEAR=vistalab/neuro-detect
docker build --tag $GEAR:$1 .
