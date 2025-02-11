% Programme to calculate the wavefront from a given Hartmanngram, assuming
% that the unknown reference wavefront is given by the best fit of the spots.
% Then, use the Beam Propagation Method (BPM - 3D) based on K. Okamoto Ch 7.
% All units are in microns.
% Authors: Masoud, Mabel, Barbara, Erez, 2020-2023 (Updated 02-2025)

close all; clearvars;

%% 1. Parameters
mm = 1e3; % millimeters to microns
um = 1;   % microns
nm = 1e-3; % nanometers to microns

% Physical parameters
l = 45 * mm; % Distance between Hartmann plate and camera
pixel = 13.5 * um; % Camera pixel size
pitch = 150 * um; % Hartmann holes pitch
wl = 13.5 * nm; % Wavelength of light
thrsh = 0.05; % Noise threshold for intensity
thr = 0.05; % Mask threshold
limit = 0; % Noise threshold after smoothing
cl = pitch / (2 * pi * l); % Calibration factor to convert angles to distances

%% 2. Search the file
[filename, dirPath] = uigetfile({'*.SPE;*.txt'}, 'Select a file', 'C:\Users\...');
disp(filename);
ft = char(split(filename, '.')); 
ft = ft(2, 1:3); % File extension type

% Read the Hartmanngram file
if ft == 'SPE'
    hf = double(readSPE(dirPath, filename)); % For SPE files
elseif ft == 'txt'
    hf = importdata([dirPath filename]); % For text files
end

if isstruct(hf)
    hf = hf.data; % Extract data if it's a structure
end

% Read the reference file
hr = double(readSPE('7min_2um_0dgr2.SPE')); % Built-in reference folder

%% 3. Magnification reference file
[Ny, Nx] = size(hf); % Size of the Hartmanngram
N21 = Nx / 2 + 1; % Center of the array

% Resize and pad the reference file
hr = padarray(imresize(hr, 0.5), [Ny/4, Nx/4], 'both');
hg = imresize(hr, 0.5);
hr = zeros(Ny, Nx);
hr(1:Ny/2, 1:Nx/2) = hg;
hr = circshift(hr, [Ny/4, Nx/4]);

% Fourier transform of the reference file
HR = fftshift(fft2(hr));

% Find peaks in the Fourier transform to determine magnification
hprofile = abs(HR(N21, :)); % Horizontal profile
vprofile = abs(HR(:, N21)); % Vertical profile

[hpeaks, hpos] = findpeaks(hprofile, 'MinPeakDistance', Nx/40, 'MinPeakHeight', max(hprofile)/11);
[vpeaks, vpos] = findpeaks(vprofile, 'MinPeakDistance', Ny/40, 'MinPeakHeight', max(vprofile)/11);

qx = N21 - hpos(ceil(length(hpeaks)/2) - 1); % Distance between central peak and first lobe (horizontal)
qy = N21 - vpos(ceil(length(vpeaks)/2) - 1); % Distance between central peak and first lobe (vertical)
qr = mean([qx, qy]); % Average distance
qq = round(qr); % Rounded distance

%% 4. Calculating intensity
hf = hf / max(hf(:)); % Normalize intensity to 1
hf(hf < thrsh) = 0; % Remove noise below the threshold

% Resize and pad the Hartmanngram
hf = padarray(imresize(hf, 0.5), [Ny/4, Nx/4], 'both');

% Display the Hartmanngram and draw an ellipse to define the region of interest
fig = uifigure;
fig.Position = [1396 105 480 170]; % Create a centered smaller mask
figure(1);
set(1, 'pos', [20 75 1359 1023]);
imagesc(hf);
axis image;
colormap("jet");
colorbar;

h = drawellipse('Center', [N21, N21], 'SemiAxes', 0.5 * [N21, N21], 'StripeColor', 'r');
uialert(fig, 'Reshape, rotate, click here', 'Program Information', 'Icon', 'info', 'CloseFcn', 'uiresume(fig)');
uiwait(fig);

hc = round(h.Center); % Center of the ellipse
cr = circshift(createMask(h), hc - [N21, N21]); % Create a mask from the ellipse
close(fig);

% Shift the Hartmanngram to the center
hf = circshift(hf, hc - [N21, N21]);
figure(1);
clf;
imagesc(hf);
axis image;
colorbar;

% Fourier transform of the Hartmanngram
HF = fftshift(fft2(hf));

% Find peaks in the Fourier transform to determine magnification
hprofile = abs(HF(N21, :));
vprofile = abs(HF(:, N21));

[hpeaks, hpos] = findpeaks(hprofile, 'MinPeakDistance', Nx/40, 'MinPeakHeight', max(hprofile)/11);
[vpeaks, vpos] = findpeaks(vprofile, 'MinPeakDistance', Ny/40, 'MinPeakHeight', max(vprofile)/11);

qx = N21 - hpos(ceil(length(hpeaks)/2) - 1);
qy = N21 - vpos(ceil(length(vpeaks)/2) - 1);
qz = mean([qx, qy]);
qq = round(qz);

qbx = mean(diff(hpos));
qby = mean(diff(vpos));
Q1 = [qbx, qby];
qbq = mean(Q1);

ref_q = 2 * (Nx / (pitch / pixel)); % Mean magnification for the ideal planar wavefront
mag_d = qbq / ref_q;
focdis = l * 1e-6 / (1 / mag_d - 1); % Focal distance
dis_Hart_source = sprintf('%.5f m', focdis); % Display the focal distance

%% 4.2. Creating a mask
ptch = round(pitch / pixel / 2 * 2.5); % Array shrinking leads to twice smaller pitch
m = imclose(hf > thr, strel('disk', ptch + 1)); % Define mask in HS aperture foci
m = imopen(m, strel('disk', ptch - 1));
m = m .* cr; % Apply the ellipse mask

n = 1 - m; % Anti-mask
sm = sum(m(:)); % Sum of mask pixels
sn = sum(n(:)); % Sum of anti-mask pixels

% Create circular masks for filtering
CB = CosBell(Nx, Ny, qq/8, qq/9); % Circular mask, centered
CO = fftshift(CosBell(Nx, Ny, 3.5, 1.5)); % Same, but centered on origin for Fourier plane

%% 4.3. Filtering the central lobe
HFC = HF .* CB; % Low-pass filter
hfc = ifft2(fftshift(HFC)); % Transform back to image domain
hfc(hfc < limit) = 0; % Remove noise in the retrieved intensity

% Display the results
figure(21);
set(gcf, 'pos', [5 497 976 498]);
subplot(221); imagesc(hf); axis image; colorbar; title('Input, embedded'); caxis([0 0.2]);
subplot(222); imagesc(hfc); axis image; colorbar; title('Smoothed intensity');
subplot(223); imagesc(m); axis image; colorbar; title('H-S foci mask');
subplot(224); imagesc(abs(HF)); axis image; colorbar; title('H-S transform'); caxis([0 100]);
drawnow;

%% 5. Side lobes and slopes
cl = pitch / (2 * pi * (l + focdis)); % Calibrate angles to distances

% Shift the side lobes to the center and filter
HFX = circshift(HF, [0, -qq - 1]) .* CB;
HFY = circshift(HF, [-qq - 1, 0]) .* CB;

% Transform back to image domain
hfx = ifft2(ifftshift(HFX));
hfy = ifft2(ifftshift(HFY));

% Unwrap phase from center [radians]
ax = cl * unwrapc(angle(hfx));
ay = cl * unwrapc(angle(hfy));

% Check if the aperture is segmented
cc = bwconncomp(m);
nobjcts = cc.NumObjects;

if nobjcts > 1 % Segmented aperture, iteratively fill in the data
    dx = hfx .* cr;
    dy = hfy .* cr;
    sx = sum(dx(:)) / sm;
    sy = sum(dy(:)) / sm;
    SX = sx * Nx * Ny;
    SY = sy * Nx * Ny;

    bx = sx .* n + dx .* m; % Fill angles with average data, plug in real data within mask
    by = sy .* n + dy .* m;

    stps = 40; % Number of max iterations
    tx = zeros(1, stps);
    ty = tx;

    for k = 1:stps
        t = bx;
        BX = CO .* fft2(bx);
        BX(1) = SX; % Keep phase sum constant
        bx = ifft2(BX) .* n + dx .* m; % Use a low-pass filter to fill in gaps in mask in x
        tx(k) = mean(abs(t(:) - bx(:)));
        if abs(tx(k)) < 1e-6
            break;
        end
    end

    for k = 1:stps
        t = by;
        BY = CO .* fft2(by);
        BY(1) = SY; % Keep phase sum constant
        by = ifft2(BY) .* n + dy .* m; % Use a low-pass filter to fill in gaps in mask in y
        ty(k) = mean(abs(t(:) - by(:)));
        if abs(ty(k)) < 1e-6
            break;
        end
    end

    figure(20);
    set(gcf, 'pos', [1387 42 531 436]);
    plot(tx); hold on; plot(ty); title('Slopes'' convergence'); grid; drawnow;

    ax = cl * unwrapc(angle(bx));
    ay = cl * unwrapc(angle(by));
end

% Remove average value outside segments
bx = (ax - sum(sum(ax .* m)) / sm) .* m;
by = (ay - sum(sum(ay .* m)) / sm) .* m;

% Better display background
vx = ax .* m + sum(sum(ax .* m)) / sm .* n;
vy = ay .* m + sum(sum(ay .* m)) / sm .* n;

% Obtain the phase from the slopes by integration
phasef = fft_s2p(bx, by);

%% 5.3 Analyzing the aberrations
aberrations = {'Piston', 'Tiltx', 'Tilty', 'defocus', 'astigma45', 'AstigmX', 'comaX', 'comaY', 'spherical', 'trifoilX', 'trifoilY', '5thSpherical'};
[DCf, Zernikef] = ZernikeCalc([0 0; 1 1; 1 -1; 2 0; 2 2; 2 -2; 3 1; 3 -1; 4 0; 3 -3; 3 3; 4 -2]', phasef, cr, 'standard');

% Remove Piston, Tilts, and defocus from the wavefront
phzf = (phasef - DCf(:, :, 1) - DCf(:, :, 2) - DCf(:, :, 3) - DCf(:, :, 4)) .* cr;

% Display the results
X = categorical(aberrations);
X = reordercats(X, aberrations);

figure(22);
set(gcf, 'pos', [872 10 1046 986]);
subplot(331); imagesc(ax .* cr); axis image; colorbar; title('Horizontal unwrapc angle, rad');
subplot(334); imagesc(ay .* cr); axis image; colorbar; title('Vertical unwrapc angle, rad');
subplot(332); imagesc(vx); axis image; colorbar; title('Horizontal unwrapc angle, rad');
subplot(335); imagesc(vy); axis image; colorbar; title('Vertical unwrapc angle, rad');
subplot(333); imagesc(hfc .* m); axis image; colorbar; title('Intensity');
subplot(336); imagesc(phzf .* m); axis image; colorbar; title('Full wavefront - Zer1235[mm]');
subplot(3, 3, [7, 8.9]); bar(X, Zernikef); title('Aberrations fd'); grid;


%Figures phase and intensity

% Set NaN values where the mask is zero
cr = double(cr);  % Convert from logical to double
cr(cr == 0) = NaN;  % Now you can assign NaN

% Apply the mask to the phase and intensity data
phasezf = phzf .* cr;
intensity = hfc .* cr;

% Define the clipping range and target size
clip_range = 512:1532; % Clipping range for both dimensions
target_size = [2048, 2048]; % Target size for resizing

% Clip and resize the phase and intensity data
clipped_phase = phasezf(clip_range, clip_range);
resized_phase = imresize(clipped_phase, target_size);

clipped_intensity = intensity(clip_range, clip_range);
resized_intensity = imresize(clipped_intensity, target_size);

% Function to display an image with formatted axes
function display_image(image, title_text)
    figure;
    imagesc(image);
    axis image;
    colormap("jet"); % Use a colormap for better visualization
    colorbar;
    
    % Format tick labels to show 0 to 15.2 with one decimal place
    xticks(linspace(0, size(image, 2), 6));
    xticklabels(num2str(linspace(0, 15.2, 6)', '%.1f'));
    yticks(linspace(0, size(image, 1), 6));
    yticklabels(num2str(linspace(0, 15.2, 6)', '%.1f'));
    
    xlabel('mm');
    ylabel('mm');
    title(title_text);
end

% Display the resized intensity and phase images
display_image(resized_intensity, 'Resized Intensity');
display_image(resized_phase, 'Resized Phase');
