#! /usr/bin/env python

# Parse a config file and create a dtiInit params json file.
def parse_config(input_file, output_file, input_dir, output_dir, nifti_dir, bvec_dir, bval_dir, anat_dir):
    import os
    import json
    import glob
    import shutil

    if not os.path.isfile(input_file):
        manifest = "/flywheel/v0/manifest.json"
        input_file = manifest
        MANIFEST=True
    else:
        MANIFEST=False

    # Read the config json file
    with open(input_file, 'r') as jsonfile:
        config_json = json.load(jsonfile)

    if MANIFEST:
        print("Loading default configuration from %s" % input_file)
        manifest_config = dict.fromkeys(config_json['config'].keys())
        for k in manifest_config.iterkeys():
            manifest_config[k] = config_json['config'][k]['default']
        config = dict()
        config['params'] = manifest_config
    else:
        # Rename the config key to params
        print("Parsing %s" % input_file)
        config = dict()
        if config_json['config'].has_key('config'):
            config['params'] = config_json['config']['config']
        else:
            config['params'] = config_json['config']

    # Combine to build the dwOutMm array ( This can be removed once support for arrays is added in the schema. )
    dwOutMm = [config['params']['dwOutMm-1'], config['params']['dwOutMm-2'], config['params']['dwOutMm-3']]
    config['params']['dwOutMm'] = dwOutMm

    # Remove the other dwOutMm fields
    del config['params']['dwOutMm-1']
    del config['params']['dwOutMm-2']
    del config['params']['dwOutMm-3']

    # Add input directory for dtiInit
    config['input_dir']  = input_dir
    config['output_dir'] = output_dir

    # Add anatomical file for dtiInit
    if os.path.exists(anat_dir) and os.listdir(anat_dir):
        config['t1_file'] = glob.glob(os.path.join(anat_dir, '*.nii.gz'))[0]
    else:
        config['t1_file'] = '/templates/MNI_EPI.nii.gz'


    # Rename the bval and bvec files to match the dti file to deal with how the
    # matlab code combines related files
    config['bval_file'] = glob.glob(os.path.join(bval_dir, '*.bval*'))[0]
    config['bvec_file'] = glob.glob(os.path.join(bvec_dir, '*.bvec*'))[0]
    config['dwi_file']  = glob.glob(os.path.join(nifti_dir, '*.nii.gz'))[0]

    # Copy dwi nifti file to work dir to be processed
    shutil.copyfile(config['dwi_file'], os.path.join(input_dir, os.path.basename(config['dwi_file'])))

    # Dervive base names w/o extensions for comparission
    dwi_name  = config['dwi_file'].split('.')[0].split('/')[-1]
    bval_name = config['bval_file'].split('.')[0].split('/')[-1]
    bvec_name = config['bvec_file'].split('.')[0].split('/')[-1]

    # If either the bval or bvec file does not share the same basename as the
    # dwi file, make a copy of that file with the correct basename
    if bval_name is not dwi_name:
        shutil.copyfile(config['bval_file'], os.path.join(input_dir, dwi_name + '.bval'))
    else:
        shutil.copyfile(config['bval_file'], os.path.join(input_dir, bval_name + '.bval'))

    if bvec_name is not dwi_name:
        shutil.copyfile(config['bvec_file'], os.path.join(input_dir, dwi_name + '.bvec'))
    else:
        shutil.copyfile(config['bvec_file'], os.path.join(input_dir, bvec_name + '.bvec'))

    # Write out the modified configuration
    with open(output_file, 'w') as config_json:
        json.dump(config, config_json)

if __name__ == '__main__':

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file', default='/flwywheel/v0/config.json', help='Full path to the input file.')
    ap.add_argument('--output_file', default='/flywheel/v0/dtiInit.json', help='Full path to the output file.')
    ap.add_argument('--input_dir', default='/flwywheel/v0/input', help='Full path to the input file.')
    ap.add_argument('--output_dir', default='/flywheel/v0/output', help='Full path to the output file.')
    ap.add_argument('--bvec_dir', default='/flywheel/v0/input/bvec', help='Full path to the bvec directory.')
    ap.add_argument('--bval_dir', default='/flywheel/v0/input/bval', help='Full path to the bval directory.')
    ap.add_argument('--nifti_dir', default='/flywheel/v0/input/dwi', help='Full path to the nifti directory.')
    ap.add_argument('--anat_dir', default='/flywheel/v0/input/anatomical', help='Full path to the anatomical directory.')
    args = ap.parse_args()

    parse_config(args.input_file, args.output_file, args.input_dir, args.output_dir, args.nifti_dir, args.bvec_dir, args.bval_dir, args.anat_dir)
