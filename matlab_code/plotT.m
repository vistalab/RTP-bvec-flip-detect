function [] = plotT(tensor)
    [x,y,z] = sphere(15);
    sz = size(x);
    u = [x(:), y(:), z(:)];
    D = dt6VECtoMAT(tensor);
    % scale the unit vectors according to the eigensystem of D to make the ellipsoid 
    [vec,val] = eig(D);
    % 2*lambda_i*T; T=1
    e = u*(2*val)*vec';
    x = reshape(e(:,1),sz); y = reshape(e(:,2),sz); z = reshape(e(:,3),sz);
    cmap = autumn(255);
    surf(x,y,z,repmat(1,size(z)));
    axis equal, colormap([.25 .25 .25; cmap]), alpha(0.5)
    xlabel('x');ylabel('y');zlabel('z'); 
  
  end