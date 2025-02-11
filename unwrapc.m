function a = unwrapc(ar, tol)
%   UNWRAPC unwraps an array from its centre, as opposed to the Matlab
%   UNWRAP which starts from the side of the array.
%   A = UNWRAP (B) will return the unwrapped phase of array B.

% Erez Ribak Nov 2005. Start from same line Feb 2008

if nargin <2, tol = pi; end
sz = size (ar); sx = round (sz(2) / 2); sy = round (sz(1) / 2);
a1 = ar (sy:-1:1, :);
a1 = unwrap (a1, tol);
a2 = ar (sy:end, :);
a2 = unwrap (a2, tol);
a =  [a1(sy:-1:2, :);  a2];
a1 = a (:, sx:-1:1);
a1 = unwrap (a1, tol, 2);
a2 = a (:, sx:end);
a2 = unwrap (a2, tol, 2);
a =  [a1(:, sx:-1:2)  a2];
end
