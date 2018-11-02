  function tensor = getTensorFromDWI(dwi, B)
  
    % dwi = squeeze(dwi(34, 41, 40,:));
    % dwi = squeeze(dwi(29, 56, 39,:));
    % dwi = squeeze(dwi(45, 86, 42,:));
    
    bval = B(:,4);
    bvec = B(:,1:3);
    % bvecFlippedX = 



    S0   = mean(dwi((bval==0)));
    dSig = double(dwi(~(bval==0)));
    ADC  = - diag( (bval(~(bval==0))).^-1 )  *  log(dSig(:)/S0);
    b = bvec(~(bval==0),:);
    V = [b(:,1).^2, ...
         b(:,2).^2, ...
         b(:,3).^2, ...
         2 * b(:,1) .* b(:,2), ...
         2 * b(:,1) .* b(:,3), ...
         2 * b(:,2) .* b(:,3)];


     tensor = V\ADC;
  
  end