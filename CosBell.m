function cBell = CosBell(sx,sy,r,t,x0,y0)
%	COSBELL(SX,SY,R,T,X0,Y0) create a cosine bell: an array of
%	ones near the center, zeros outside it, and a smooth
%	transition in between. SY, SX are the array dimensions.
%	R is the radius of the internal region of ones. T is
%	the size of the transition in which the output changes
%   from 1 to 0 (default width 0). The outer radius is thus R+T.
%   X0, Y0 are the coordinates of the filter center (default
%   centre).

% Erez Ribak, 2007 December 11

if nargin < 3, disp ('not enough data in CosBell'); return; end
if nargin < 4,  t = 0; end
if nargin < 5, x0 = sx / 2 + 1; end
if nargin < 6, y0 = sy / 2 + 1; end
[X, Y] = meshgrid ([1:sx] - x0, [1:sy] - y0);
F = sqrt (X.^2 + Y.^2);
cBell = .5 + .5  * cos (0.5 * pi * ((F - r - t/2)/(t/2 + eps) + 1));
cBell (F <= r) = 1;
cBell (F >= r+t) = 0;
end
