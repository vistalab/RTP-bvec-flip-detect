%% Test several flips on tensor by modifying bvecs

DWI = niftiRead('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.nii.gz');
DWI = DWI.data;

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
disp("Matrix normal")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B(:,1) = -B(:,1);
disp("Matrix x flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,2) = -B(:,2);
disp("Matrix y flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,3) = -B(:,3);
disp("Matrix z flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,2) = -B(:,2);
B(:,3) = -B(:,3);
disp("Matrix y, z flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,3) = -B(:,3);
B(:,1) = -B(:,1);
disp("Matrix x, z flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,2) = -B(:,2);
B(:,1) = -B(:,1);
disp("Matrix x, y flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
B(:,3) = -B(:,3);
B(:,2) = -B(:,2);
B(:,1) = -B(:,1);
disp("Matrix x, y, z flipped")
tensor = getTensorFromDWI(dwi, B);
D = dt6VECtoMAT(tensor)