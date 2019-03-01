#!/bin/bash
# Builds the gear/container
# The container can be exported using the export.sh script
echo "You introduced version $1"
read -p "Continue (y/n)?" choice
case "$choice" in 
    y|Y ) 
        echo "Compiling Matlab, commiting the code and building the container"
        cd source/
        . ./compile.sh
        cd ../
        git add .
        git commit -m "Commiting before building $GEAR-$1"
        GEAR=vistalab/neuro-detect
        docker build --tag $GEAR:$1 . ;;
    n|N ) echo "Exiting";;
    * )   echo "invalid";;
esac

