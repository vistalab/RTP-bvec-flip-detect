%% Plot several voxels and also plot the same with inverted x axis (*-1)

DWI = niftiRead('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.nii.gz');
DWI = DWI.data;

B = dlmread('/Users/fabianreith/Documents/Matlab_experiments/data/data_aligned_trilin_noMEC.b');
 

figure(1)
subplot(2,2,1)
% voxelCorpusCallosum
dwi = squeeze(DWI(39, 44, 39,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,2)
% voxelCoords2
dwi = squeeze(DWI(34, 41, 40,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,3)
% voxelGM
dwi = squeeze(DWI(45, 86, 42,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,4)
% voxelFiberCrossing
dwi = squeeze(DWI(29, 56, 39,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

figure(2)
B(:,1) = -B(:,1);
subplot(2,2,1)
% voxelCorpusCallosum
dwi = squeeze(DWI(39, 44, 39,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,2)
% voxelCoords2
dwi = squeeze(DWI(34, 41, 40,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,3)
% voxelGM
dwi = squeeze(DWI(45, 86, 42,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)

subplot(2,2,4)
% voxelFiberCrossing
dwi = squeeze(DWI(29, 56, 39,:));
tensor = getTensorFromDWI(dwi, B);
plotT(tensor)
