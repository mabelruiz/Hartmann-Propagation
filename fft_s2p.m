function phi = fft_s2p (Sx, Sy)
% phi = fft_s2p (Sx, Sy) solves the shears - to - phase problem using periodic exp (ik*2pi/N) series
% Sx (y, x) = X_shear = phi (y, x + 1) - phi (y, x)
% Sy (y, x) = Y_shear = phi (y + 1, x) - phi (y, x)
% phi, Sx, Sy = (1:Ny, 1:Nx) 
% You may use init_s2p () for dummy Sx, Sy

[Ny, Nx] = size (Sx);
oneX = ones (1, Nx); oneY = ones (1, Ny);
% calc the fourier transform: Sx is exp (-ikx*x*2pi/Nx)*exp (-iky*y*2pi/Ny)
SQx = fft2 (Sx); SQy = fft2 (Sy);
%calc the Qg factors.: Qgx = 2*sin (Q*pi/Nx) 
QX = [0:Nx - 1]; QY = [0:Ny - 1];
Qgx = exp (i*2*pi/Nx*QX) - 1; 	% 2*sin (X*pi/Nx) if attributed to the middle;
Qgy = exp (i*2*pi/Ny*QY) - 1; 	% 2*sin (Y*pi/Ny);
Qgx (1) = eps;			                    % avoid small number division

%calc FQ = fi_Q
for qx = 1:Nx, 
    for qy = 1:Ny, 
        FQ (qy, qx) = (conj (Qgx (qx))*SQx (qy, qx) + conj (Qgy (qy))*SQy (qy, qx) )/ ( Qgx (qx)*conj (Qgx (qx)) + Qgy (qy)*conj (Qgy (qy)) );
    end ;
end ;
FQ (1, 1) = 0;
phi = real (ifft2 (FQ));  % inverse fft
