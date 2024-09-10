# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:59:41 2023

@author: simon
"""

import LightPipes as lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
import os
import warnings
import cv2
import scipy.io
import time

#path to my_functions_for_LightPipes
os.chdir("C:/Users/simon/OneDrive - Univerzita Karlova/Prace/propagace svazku/Python/Lighpipes")
import my_functions_for_LightPipes as F

#change directory to the path to this script
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

#%% initialize
#start of user parameters entries
#------------
wln = 13.5*lp.nm  # wavelength
N = 512  # grid points in one dimension (NxN is field) - choose N from (256, 512, 1024, 2048)
grid_size = 13.5*lp.um*2048*2  # size of the square grid = /
#camera pixel size*width in pixels*2 (because CCD image of Hart holes is rescaled
#to half and padded with zeroes back to 2048 pixels)
f_short = 20.2*lp.mm #distance from Hartmann plate to focus

matlab_folder = "SO_50ms_Nb400+Nb400_ap75mm_3/" #folder with amplitude and phase_in_um from Erez's MATLAB code
amplitude_mask_file_suffix = "" #"", or None, if no mask to be used. Possible masks =>
phase_mask_file_suffix = ""  # modified_1

beam_x_shift = 25.852*lp.mm #shift beam to the middle of grid (numbers e.g. from ImageJ)
beam_y_shift = 26.5*lp.mm

wait_time_show_init_beam = 0 #None = dont show, 0 = show until button press, > 0 = show image for time in ms
#------------
#end of user parameters entries

#loading amplitude and phase from file
AmplitudeMATLAB = scipy.io.loadmat(matlab_folder+"Amplitude.mat")["hfc"]
PhaseMATLAB = scipy.io.loadmat(matlab_folder+"Phase_in_um.mat")["phasef"]*lp.um*2*np.pi/wln #PhaseMATLAB is now in radians
if amplitude_mask_file_suffix:
    Amplitude_for_mask = cv2.imread(matlab_folder+f"Amplitude_16bit_{amplitude_mask_file_suffix}.tif",
                                    cv2.IMREAD_UNCHANGED)
    mask_for_MATLAB_field = [Amplitude_for_mask > 0][0]
    AmplitudeMATLAB = np.multiply(AmplitudeMATLAB, mask_for_MATLAB_field)
if phase_mask_file_suffix:
    Phase_for_mask = cv2.imread(matlab_folder+f"Amplitude_16bit_{phase_mask_file_suffix}.tif",
                                cv2.IMREAD_UNCHANGED)
    mask_for_MATLAB_field = [Amplitude_for_mask > 0][0]
    PhaseMATLAB = np.multiply(PhaseMATLAB, mask_for_MATLAB_field)


F0 = lp.Begin(grid_size, wln, AmplitudeMATLAB.shape[0]) #beam is resampled later
F0 = lp.SubIntensity(F0, AmplitudeMATLAB**2) #size is 2048
F0 = lp.SubPhase(F0, -PhaseMATLAB) #size is 2048

#shift beam to middle, and scale grid_size and N
F0 = lp.Interpol(F0, grid_size, N, x_shift = F0.grid_size/2-beam_x_shift,
                 y_shift = F0.grid_size/2-beam_y_shift)
if wait_time_show_init_beam is not None and wait_time_show_init_beam >= 0:
    F.plot_intensity_and_phase_cv2(F0, wait_time=wait_time_show_init_beam,
                                   title="Intensity and Phase at Hartmann plate")

#field in the far field of a lens
F_ff = lp.LensFarfield(F0, f_short)
print(f"Far Field grid size: {F_ff.grid_size/lp.um} um")

#position where propagation method changes
end_z_reg = f_short * (1 - F_ff.grid_size/F0.grid_size)
print(f"Z position for reg propagation for the same grid sizes: {end_z_reg/lp.mm} mm")

#Field initialized

#%% SO caustic movie
#start of user parameters entries for movie
#------------
im_size = 512 #windows image size in pixels
propagator = lp.Forvard #Fresnel, Forvard
reg_propagator = lp.Forvard_reg #Forvard_reg
wait_time = 1 #wait time between movie images (in ms), 0 means wait for button press

save_movie = False #True -> save movie, False -> only show movie
folder_movie_save = "Propagation_movies/" + matlab_folder[:-1] #path to save folder
filetype_movie_save = "png" #"png" - 8bit, "tif" - 16bit

#set number of steps for movie
#either quick by setting this one number
#or set number of steps for each propagation method below
quick_num_of_steps = 4 

num_of_steps_reg_to_focus = 2#quick_num_of_steps
distance_reg_prop_to_focus = 0.003*lp.mm #furthest propagation distance from the focus 
log_steps_reg_to_focus = False

num_of_steps_ff_to_focus = int(quick_num_of_steps / 2)
num_of_steps_ff_from_focus = int(quick_num_of_steps / 2)
closeness_to_focus = 0.01*lp.um #closest propagation distance from the focus 

num_of_steps_reg_from_focus = quick_num_of_steps
log_steps_reg_from_focus = False

#------------
#end of user parameters entries

z_pos = 0

if (f_short - distance_reg_prop_to_focus) > end_z_reg:
    raise Exception("distance_reg_prop_to_focus must be larger.")

dist_range_reg_to_focus = np.array([f_short - distance_reg_prop_to_focus,
                                    end_z_reg])
dist_range_ff_to_focus = np.array([end_z_reg - f_short,
                                   -closeness_to_focus])
dist_range_ff_from_focus = np.array([closeness_to_focus,
                                     f_short - end_z_reg])
dist_range_reg_from_focus = np.array([0,
                                      distance_reg_prop_to_focus - f_short + end_z_reg])

#properties of text in window
text_properties = {"font" : cv2.FONT_HERSHEY_SIMPLEX,
                  "bottomLeftCornerOfText" : (0,int(im_size*0.98)),
                  "fontScale" : 0.4,
                  "fontColor" : (255,255,255),
                  "thickness" : 1,
                  "lineType" : 2
                  }

#initialize windows
window_name_int = "Intensity"
window_name_phase = "Phase"
if not save_movie:
    cv2.namedWindow(window_name_int)
    cv2.moveWindow(window_name_int, 0, 0)
    cv2.namedWindow(window_name_phase)
    cv2.moveWindow(window_name_phase, im_size+1, 0)

z_list = []
Aeff_list = []
D4sigmas_list = []
sharpness_list = []


#propagate regularized to distance that is close to focus
F_step, z_pos, [tmp_z_list, tmp_Aeff_list, tmp_D4sigmas_list, tmp_sharpness_list] = \
    F.movie_propagate(save_movie, F0, reg_propagator, num_of_steps_reg_to_focus,
                      dist_range_reg_to_focus, z_pos,
                      text_properties, [window_name_int, window_name_phase],
                      wait_time, im_size = im_size, f_lens = f_short,
                      substract_defocus=False, log_steps = log_steps_reg_to_focus,
                      folder = folder_movie_save, filetype = filetype_movie_save)
z_list.extend(tmp_z_list)
Aeff_list.extend(tmp_Aeff_list)
D4sigmas_list.extend(tmp_D4sigmas_list)
sharpness_list.extend(tmp_sharpness_list)

#propagate to focus from field created by far field
z_pos = f_short #I propagate from focus to both directions

F_step, z_pos, [tmp_z_list, tmp_Aeff_list, tmp_D4sigmas_list, tmp_sharpness_list] = \
    F.movie_propagate(save_movie, F_ff, propagator, num_of_steps_ff_to_focus,
                      dist_range_ff_to_focus, z_pos,
                      text_properties, [window_name_int, window_name_phase],
                      wait_time, im_size = im_size, include_first = True,
                      folder = folder_movie_save, filetype = filetype_movie_save)
    
z_list.extend(tmp_z_list)
Aeff_list.extend(tmp_Aeff_list)
D4sigmas_list.extend(tmp_D4sigmas_list)
sharpness_list.extend(tmp_sharpness_list)

#focus as a far field
F_step, z_pos, [tmp_z_list, tmp_Aeff_list, tmp_D4sigmas_list, tmp_sharpness_list] = \
    F.movie_propagate(save_movie, F0, reg_propagator, 1, [0, f_short], 0,
                      text_properties, [window_name_int, window_name_phase],
                      wait_time, im_size = im_size, f_lens = f_short,
                      substract_defocus=False, include_first = False,
                      folder = folder_movie_save, filetype = filetype_movie_save)

z_list.extend(tmp_z_list)
Aeff_list.extend(tmp_Aeff_list)
D4sigmas_list.extend(tmp_D4sigmas_list)
sharpness_list.extend(tmp_sharpness_list)

#propagate from focus from field created by far field
F_step, z_pos, [tmp_z_list, tmp_Aeff_list, tmp_D4sigmas_list, tmp_sharpness_list] = \
    F.movie_propagate(save_movie, F_ff, propagator, num_of_steps_ff_from_focus,
                      dist_range_ff_from_focus, z_pos,
                      text_properties, [window_name_int, window_name_phase],
                      wait_time, im_size = im_size, include_first = True,
                      folder = folder_movie_save, filetype = filetype_movie_save)
    
z_list.extend(tmp_z_list)
Aeff_list.extend(tmp_Aeff_list)
D4sigmas_list.extend(tmp_D4sigmas_list)
sharpness_list.extend(tmp_sharpness_list)

#propagate regularized from distance that is close to focus
F_step, z_pos, [tmp_z_list, tmp_Aeff_list, tmp_D4sigmas_list, tmp_sharpness_list] = \
    F.movie_propagate(save_movie, F_step, reg_propagator, num_of_steps_reg_from_focus,
                      dist_range_reg_from_focus, z_pos,
                      text_properties, [window_name_int, window_name_phase],
                      wait_time, im_size = im_size, f_lens = -(f_short - end_z_reg),
                      substract_defocus=True, log_steps = log_steps_reg_from_focus,
                      folder = folder_movie_save, filetype = filetype_movie_save)
z_list.extend(tmp_z_list)
Aeff_list.extend(tmp_Aeff_list)
D4sigmas_list.extend(tmp_D4sigmas_list)
sharpness_list.extend(tmp_sharpness_list)

#close windows
if not save_movie:
    cv2.destroyWindow("Intensity")
    cv2.destroyWindow("Phase")


diff_from_focus_um = (np.array(z_list) - f_short)/lp.um

fig, ax1 = plt.subplots()
ax1.plot(diff_from_focus_um,np.array(Aeff_list)/lp.um**2, label="Aeff")
ax1.plot(diff_from_focus_um,((np.array(D4sigmas_list).T[0] * np.array(D4sigmas_list).T[1]) /
                 16/lp.um**2*np.pi/2),
         label=r"$\pi\sigma_x\sigma_y/2$")
#ax1.plot(diff_from_focus_um,((np.array(D4sigmas_list).T[1])/4)**2/lp.um**2, label=r"$\sigma_y^2$")
ax1.plot(np.array([end_z_reg-f_short, end_z_reg-f_short])/lp.um,
         [0, max(Aeff_list)/lp.um**2], linestyle=":")
ax1.plot(np.array([1*f_short-end_z_reg, 1*f_short-end_z_reg])/lp.um,
         [0, max(Aeff_list)/lp.um**2], linestyle=":")
ax1.set_xlabel(f"Distance from focus at {f_short/lp.mm} mm [um]")
ax1.set_ylabel("Beam dimensions [um2]")
plt.legend()
# =============================================================================
# ax2 = ax1.twinx()
# ax2.plot(diff_from_focus_um,sharpness_list, label="sharpness", color = "red")
# =============================================================================
#ax.set_xlim([-120,-20])
#ax.set_ylim([None,None])
plt.legend()
if save_movie:
    plt.savefig(f"{folder_movie_save}/_Plot_size_vs_position.png")
else:
    plt.show()
    
if save_movie:
    F.merge_folders(folder_movie_save)
