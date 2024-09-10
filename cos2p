function phi = cos_s2p (Sx, Sy)
%function phi = cos_s2p (Sx, Sy, Nx, Ny)
% phi = cos_s2p (Sx, Sy, Nx, Ny) solves exactly the shears - to - phase problem with Cos series
% Sx (y, x) = X_shear = phi (y, x + 1) - phi (y, x)
% Sy (y, x) = Y_shear = phi (y + 1, x) - phi (y, x)
% phi, Sx, Sy = (1:Ny, 1:Nx) Sx (y, Nx) = Sy (Ny, x) = 0
% You may use init_s2p () for dummy Sx, Sy

% To calculate the Sine Transform, we use FFT of 2N points
[Ny, Nx] = size (Sx);
oneX = ones (1, 2*Nx); oneY = ones (1, 2*Ny);

% calc the fourier transform: Sx is sin (x) cos (y)!!
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tQy = [0:pi/Ny/2:pi]; tQy = exp (-i*tQy); ttqy = tQy.' ; 
tQx = [0:pi/Nx/2:pi]; tQx = exp (-i*tQx);
QX = [2:Nx]; QY = [2:Ny];
cQX = 2 + 2*Nx - QX; cQY = 2 + 2*Ny - QY;
toneY = oneY';
K = 2i/Nx/Ny;

work = zeros (2*Ny, 2*Nx);
work (1:Ny, QX) = Sx (1:Ny, QX - 1); 
work (1:Ny, cQX) = -work (1:Ny, QX);		% work (2 + b, y) = work (2Nx - b, y) = Sx (1 + b, y)
gw = fft2 (work);				% gw (qy) = sum{ exp[ (qy - 1) (y - 1)2pi/2Ny]*F (y)

%Fourier Cos (y) sin (x) transform of Sx
SQx (QY, QX) = (0.5*K)* (gw (QY, QX).* (ttqy (QY)*oneX (QX)) - gw (cQY, QX).* (ttqy (cQY)*oneX (QX)) );
SQx (1, QX) = K*gw (1, QX);
SQx (1:Ny, 1) = 0;

work = zeros (2*Ny, 2*Nx);
work (QY, 1:Nx) = Sy (QY - 1, 1:Nx);
work (cQY, 1:Nx) = - work (QY, 1:Nx);			%anti - symmetry in y
gw = fft2 (work);

SQy (QY, QX) = (0.5*K)* (gw (QY, QX).* (toneY (QY)*tQx (QX)) - gw (QY, cQX).* (toneY (QY)*tQx (cQX)) );
SQy (QY, 1 ) = K*gw (QY, 1);
SQy (1, 1:Nx) = 0;

%calc the Qg factors: Qgx = 2*sin (Q*pi/2/Nx) = 2*imag (tQx (Q - 1))
Qgx = -2*imag (tQx); Qgy = -2*imag (tQy);
Qgx (1) = 1e-50;

%calc FQ = fi_Q
for x = 1:Nx, 
    for y = 1:Ny, 
        FQ (y, x) = - (Qgx(x) * SQx(y, x) + Qgy(y) * SQy(y, x) )/ (Qgx(x)^2 + Qgy(y)^2);
    end ;
end ;
FQ (1, 1) = 0;

% inverse FFT
work = zeros (2*Ny, 2*Nx);
X1 = [1:Nx]; Y1 = [1:Ny];
work (Y1, X1) = FQ (Y1, X1).* (ttqy (Y1)*tQx (X1));
work (cQY, X1) = FQ (QY, X1).* (conj (ttqy (QY))*tQx (X1));
work (Y1, cQX) = FQ (Y1, QX).* (ttqy (Y1)*conj (tQx (QX)));
work (cQY, cQX) = FQ (QY, QX).* (conj (ttqy (QY))*conj (tQx (QX)));
%Note: work (Ny + 1, :) = work (:, Nx + 1) = 0 !!
gw =  real (fft2 (work));
phi (Y1, X1) = 0.25*gw (Y1, X1);
