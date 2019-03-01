#! /bin/bash 

# --entrypoint=/bin/bash \
docker run --rm -ti \
	-v /black/localhome/glerma/TESTDATA/bvecflipdetect/input:/flywheel/v0/input \
	-v /black/localhome/glerma/TESTDATA/bvecflipdetect/output:/flywheel/v0/output \
	-v /data/localhome/glerma/soft/neuro_detect/docker_dti_init_neuro_detect/example_config.json:/flywheel/v0/config.json vistalab/neuro-detect:0.4.1
