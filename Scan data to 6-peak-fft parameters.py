# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:00:56 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

def remove_nan_elements(array1):
    array2 = []
    for element in array1:
        if np.isnan(element) == False:
            array2.append(element)
    return(array2)


def generate_xyz_arrays(data_f):
    x = data_f.loc[:,0]
    y = data_f.loc[:,1]
    z = data_f.loc[:,2]
   

    x1 = np.asarray(x)
    y1 = np.asarray(y)
    z1 = np.asarray(z)
   

    xdata_f = remove_nan_elements(x1)
    ydata_f = remove_nan_elements(y1)
    zdata_f = remove_nan_elements(z1)
    
    return xdata_f, ydata_f, zdata_f


def twoD_to_1D(array_2D):
    array_op = np.asarray(array_2D)
    length = int(len(array_op)**2)
    array_op = np.reshape(array_op, (1, length))
    array_op = array_op[0]
    return array_op

def twoD_r_to_1D(array_2D):
    array_op = np.asarray(array_2D)
    length = int(np.shape(array_op)[0]*np.shape(array_op)[1])
    array_op = np.reshape(array_op, (1, length))
    array_op = array_op[0]
    return array_op
    
def oneD_to_2D(array_1D):
    length = int(len(array_1D)**0.5)
    array_op = np.reshape(array_1D, (length,length))
    array_op = np.asarray(array_op)
    return array_op

def distance(x, y):                                      #x and y are array of length 2
    r_sq = ((y[1]-y[0])**2) + ((x[1]-x[0])**2)
    r = r_sq**0.5
    return r

def Rotate_meshgrid(x_array, y_array, Angle):
    Rad = np.pi*Angle/180.
        
    Rot_Op = np.array([[np.cos(Rad), np.sin(Rad)], [-np.sin(Rad), np.cos(Rad)]])
    
    X,Y = np.meshgrid(x_array, y_array)
    
    X_rot = np.zeros(np.shape(X))
    Y_rot = np.zeros(np.shape(Y))
    count = 0
    for i in range(len(x_array)):
        for j in range(len(y_array)):
            X_rot[i,j], Y_rot[i,j] = np.dot(Rot_Op, [X[i,j], Y[i,j]])
            count = count+1
    return X_rot, Y_rot


# Generic peak filtering away from the origin.
def quadrant_1(index, peaks, x_peak_array_f, y_peak_array_f):
    value = False
    for i in range(len(index)):
        if abs(x_peak_array_f[peaks][index[i]]) > 1 or abs(y_peak_array_f[peaks][index[i]]) > 1:
            value = True
    return value


#Detect the most prominent peak in any of the quadrant.¶
def quadrant_2(index, peaks, x_fft_rot_1D, y_fft_rot_1D):
    value = False
    for i in range(len(index)):
        if abs(x_fft_rot_1D[peaks][index[i]]) > 1 and abs(y_fft_rot_1D[peaks][index[i]]) > 1:
            value = True
    return value


# Detect 3rd peak pair. Samples point in the complimentary quadrant.

def quadrant_3(index, peaks, x_ref_peak, y_ref_peak, x_fft_rot_1D, y_fft_rot_1D):
    value = False
    
    for i in range(len(index)):
        x_ref_sign = x_ref_peak/np.abs(x_ref_peak)
        y_ref_sign = y_ref_peak/np.abs(y_ref_peak)
        ref_quadrant = x_ref_sign * y_ref_sign    # TO know the quandrant of the 2nd peak
        
        y_sign = y_fft_rot_1D[peaks][index[i]]/np.abs(y_fft_rot_1D[peaks][index[i]])
        x_sign = x_fft_rot_1D[peaks][index[i]]/np.abs(x_fft_rot_1D[peaks][index[i]])
        peak_quadrant = x_sign * y_sign  # TO know the quadrant of the peak under consideration
        
        
        # The quandrant of the 3rd peak should be shifted by 90 deg in referance to the previous peak quadrant.
        for i in range(len(index)):
            if abs(x_fft_rot_1D[peaks][index[i]]) > 1 and abs(y_fft_rot_1D[peaks][index[i]]) > 1 and peak_quadrant == - ref_quadrant:
                value = True
    return value


def fft_peak_variance(file_string, image_divisions, pixels):
    
    norm_factors = []
    fft_data = []
    
    read_y_pixels = int(pixels/image_divisions)
    
    for j in range(image_divisions):   # Reads data and plots for each of the divisions.          
        
        rows_skip = 2+(pixels+1)*read_y_pixels*j
        rows_read = (pixels+1)*read_y_pixels
       
        data = pd.read_csv(file_string, delimiter = "\t", skiprows = rows_skip, nrows = rows_read, header = None)
        xdata, ydata, zdata = generate_xyz_arrays(data)

        if j == 0:    
            ydata_reduced = ydata       # Retain the reduced ydata set for fft plots.

           
        z_2D = np.reshape(zdata, (pixels, read_y_pixels))    
        fft_op = np.fft.fft2(z_2D)
        fft_op = np.fft.fftshift(fft_op)
        fft_op = np.transpose(fft_op)
        fft_array = twoD_r_to_1D(fft_op)

        norm_factor = np.max(abs(fft_array))
        fft_array = fft_array/norm_factor
        fft_data.append(abs(fft_array))

        norm_factors.append(norm_factor)    # Note: Norm factor differs for each of the fragmented image sets.
        
    frame_width_x = (np.max(xdata)-np.min(xdata))
    frame_width_y = (np.max(ydata_reduced)-np.min(ydata_reduced))

    frame_half_x = frame_width_x/2
    frame_half_y = frame_width_y/2
    
    #Define valid pixels for each of the axis
    pixels_x = pixels
    pixels_y = read_y_pixels

    #The x-inv and y-inv array in the fft space varies accordingly
    x_peak_array = (pixels_x/(frame_width_x*2))*(1/frame_half_x)*(xdata-frame_half_x*np.ones(len(xdata)))
    y_peak_array =(pixels_y/(frame_width_y*2))*(1/frame_half_y)*(ydata_reduced-frame_half_y*np.ones(len(ydata)))
    
    peak_heights_array = []
    lattice_param_array = []
    abs_height_array = []
    
    for i in range(len(fft_data)):
        
        if norm_factors[i] > 0:
            
            peaks, properties = find_peaks(fft_data[i], height = 0.05, width = 1, threshold = 0.01 )
            rounded_heights = []
        
            for element in (fft_data[i][peaks]):
                rounded_heights.append(round(element, 4))
                
            reduced_heights_set = (fft_data[i][peaks])[0:int(len(fft_data[i][peaks])/2)]
            sorted_peaks = np.asarray(sorted(reduced_heights_set, reverse = True))

            index_params = []
            lattice_params = []
            

            for element in sorted_peaks:
                index_element = np.where(rounded_heights == (round(element, 4)))[0]
                #print(index_element)

                if len(index_element) == 2 and quadrant_1(index_element, peaks, x_peak_array, y_peak_array) == True:
                    lattice_parameter = 2/(distance(x_peak_array[peaks][index_element],y_peak_array[peaks][index_element]))

                    if lattice_parameter <= 0.3 and lattice_parameter >= 0.15:

                        index_params.append(index_element)
                        lattice_params.append(lattice_parameter)
                    
            if len(index_params) > 0:
                peak_heights_array.append((fft_data[i])[peaks][index_params[0][0]])
                lattice_param_array.append(lattice_params[0])
                abs_height_array.append((fft_data[i])[peaks][index_params[0][0]]*norm_factors[i])
            
            else:
                peak_heights_array.append(0)
                lattice_param_array.append(0)
                abs_height_array.append(0)
                
        else:
            peak_heights_array.append(-2)
            lattice_param_array.append(-2)
            abs_height_array.append(-2)
            
    return abs_height_array, np.mean(abs_height_array), np.var(peak_heights_array), np.var(abs_height_array) 







def hexagon_peak_find(fft_data_f, pixels, frame_width_f, norm_coeff, xdata_f, ydata_f):
    
    frame_half = frame_width_f/2
    
    x_peak_array = (pixels/(frame_width_f*2))*(1/frame_half)*(xdata_f-frame_half*np.ones(len(xdata_f)))
    y_peak_array =(pixels/(frame_width_f*2))*(1/frame_half)*(ydata_f-frame_half*np.ones(len(ydata_f)))

    y_2d = (oneD_to_2D(y_peak_array))
    x_fft_vector = x_peak_array[0:pixels]
    y_fft_vector = y_2d[:,0]
        
      
    if norm_coeff > 0:
        
        peaks, properties = find_peaks(fft_data_f, height = 0.05, width = 0.5, threshold = 0.01 )
        
        # Sort all the peaks in the order of prominence.

        rounded_prominence = []

        for element in (properties["prominences"]):
            rounded_prominence.append(round(element, 4))

        prominent_peaks = properties["prominences"][0:int(len(properties["prominences"])/2)]

        sorted_peaks = np.asarray(sorted(prominent_peaks, reverse = True))



        # Index parameters for the 1st set of peaks.
        six_peaks_indices = []
        six_peaks_lattice = []

        index_params = []
        lattice_param_array = []

        for element in sorted_peaks:
            index_element = np.where(rounded_prominence == (round(element, 4)))[0]
            #print(index_element)

            #Filter 1 : This filters the peaks based on the location of the peak in the fourier space
            if len(index_element) == 2 and quadrant_1(index_element, peaks, x_peak_array, y_peak_array) == True:
                lattice_parameter = 2/(distance(x_peak_array[peaks][index_element],y_peak_array[peaks][index_element]))


                #Filter 2 : This filters the peaks based on lattice parameter
                if lattice_parameter <= 0.3 and lattice_parameter >= 0.15:
                    index_params.append(index_element)
                    lattice_param_array.append(lattice_parameter)

        #Index of the first peak
        if len(index_params) > 0:
            six_peaks_indices.append(index_params[0][0])
            six_peaks_indices.append(index_params[0][1])
            six_peaks_lattice.append(lattice_param_array[0])


        
        
        # Rotate the fft xy meshgrid to set the peak-1 axis to x = 0

        theta_1 =  (y_peak_array[peaks][index_params[0]][0])/(x_peak_array[peaks][index_params[0]][0])
        theta_2 =  (y_peak_array[peaks][index_params[0]][1])/(x_peak_array[peaks][index_params[0]][1])

        deg_1 = np.arctan(theta_1)*(180/np.pi)
        deg_2 = np.arctan(theta_2)*(180/np.pi)
        
        
        x_fft_rot, y_fft_rot = Rotate_meshgrid(x_fft_vector, y_fft_vector, deg_1)

        #Meshgrid basis after rotation

        x_fft_rot_1D = twoD_to_1D(x_fft_rot)
        y_fft_rot_1D = twoD_to_1D(y_fft_rot)  # Not necessary to transpose this



        
        
        # Index parameters for the 2nd set of peaks.

        index_params = []
        lattice_param_array = []

        for element in sorted_peaks:
            index_element = np.where(rounded_prominence == (round(element, 4)))[0]

            # Finds peaks in one of the quadrant.
            if len(index_element) == 2 and quadrant_2(index_element, peaks, x_fft_rot_1D, y_fft_rot_1D) == True:
                lattice_parameter = 2/(distance(x_fft_rot_1D[peaks][index_element],y_fft_rot_1D[peaks][index_element]))

                if lattice_parameter <= 0.3 and lattice_parameter >= 0.15:
                    index_params.append(index_element)
                    lattice_param_array.append(lattice_parameter)

        if len(index_params) > 0:
            six_peaks_indices.append(index_params[0][0])
            six_peaks_indices.append(index_params[0][1])
            six_peaks_lattice.append(lattice_param_array[0])
        
            x_2nd_peak = x_fft_rot_1D[peaks][index_params[0]][0]
            y_2nd_peak = y_fft_rot_1D[peaks][index_params[0]][0]

            #print(x_2nd_peak, y_2nd_peak)
        
        
        # Index parameters for the 3rd set of peaks.

        index_params = []
        lattice_param_array = []

        for element in sorted_peaks:
            index_element = np.where(rounded_prominence == (round(element, 4)))[0]
            #print(index_element)

            if len(index_element) == 2 and quadrant_3(index_element, peaks, x_2nd_peak, y_2nd_peak, x_fft_rot_1D, y_fft_rot_1D) == True:
                lattice_parameter = 2/(distance(x_fft_rot_1D[peaks][index_element],y_fft_rot_1D[peaks][index_element]))

                if lattice_parameter <= 0.3 and lattice_parameter >= 0.15:
                    index_params.append(index_element)
                    lattice_param_array.append(lattice_parameter)
        

        if len(index_params) > 0:
            six_peaks_indices.append(index_params[0][0])
            six_peaks_indices.append(index_params[0][1])
            six_peaks_lattice.append(lattice_param_array[0])
                     
            
            
                        
        
        if len(six_peaks_indices)>0:
            
            height_sum = 0
            peak_indices = peaks[six_peaks_indices]
            peak_heights = (fft_data_f)[peak_indices]
            
            for j in range(len(six_peaks_indices)):
                height_sum += peak_heights[j]
                #print(six_peaks_indices[i])

            height_avg = height_sum/len(six_peaks_indices)
            abs_avg_height = height_avg*norm_coeff
            
        else:
            height_avg = 0
            abs_avg_height = height_avg*norm_coeff
        
                   
    
    else:
        height_avg = -2
        abs_avg_height = -2
        
    return peak_indices, peak_heights, six_peaks_lattice, height_avg, abs_avg_height


# This function returns:
#(peak indices array of fft data, norm_heights array of the peaks, lattice paramter array, average height, absolute Avg. height)

def fft_parameters(file_string):
    
    os.chdir(r"C:\Users\Administrator\Anaconda\STM_Scripts\Ganesh\scripts for labview\files\BO_Experiments")
    file_name = str(file_string)
    
    pixels = 256
    rows_skip = (pixels**2+(pixels-1)+2)
    rows_read = (pixels**2+(pixels-1))
    #data_sets = int((len(pd.read_csv(file_name, delimiter = "\t", skiprows = 2, header = None))+2)/rows_skip)
    #print(data_sets)
    
    setpoint_list = []
    bias_list = []
    
    #print(frame_half)
    
    #print(xdata)
    #plt.figure(figsize = (20,40))
    
    fft_data = []
    
    i=0
    
    data_param = pd.read_csv(file_name, delimiter = "\t", skiprows = 1, nrows = 1, header = None)
    data = pd.read_csv(file_name, delimiter = "\t", skiprows = 2, header = None)
    xdata, ydata, zdata = generate_xyz_arrays(data)
    
    setpoint = data_param.loc[:,0][0]
    setpoint_list.append(setpoint)
    bias = data_param.loc[:,1][0]
    bias_list.append(bias)
    
    z_length=int(len(zdata)**0.5)
    z_2D = np.reshape(zdata, (z_length,z_length))
    fft_op = np.fft.fft2(z_2D)
    fft_op = np.fft.fftshift(fft_op)
    fft_op = np.transpose(fft_op)
    fft_array = twoD_to_1D(fft_op)
    
    norm_factor = np.max(abs(fft_array))
    fft_array = fft_array/norm_factor
    fft_data.append(abs(fft_array))
    
    
    frame_width = (np.max(xdata)-np.min(xdata))
    print(frame_width)
    frame_half = frame_width/2
    
    #FFT peak processing
    
    
    
    
    
    
    #x_peak_array = (pixels/(frame_width*2))*(1/frame_half)*(xdata-frame_half*np.ones(len(xdata)))
    #y_peak_array =(pixels/(frame_width*2))*(1/frame_half)*(ydata-frame_half*np.ones(len(ydata)))
    

    peak_height_array = []

    
    
    #plt.figure(figsize = (20,40))
    
        
    if norm_factor > 1:
    
        '''
        #print(norm_factors[i])
        peaks, properties = find_peaks(fft_data[i], height = 0.05, width = 1, threshold = 0.02 )
        print(f"total points for {i+1} scan = {len(peaks)/2}")
    
        prominence_array = []
        lattice_param_array = []
        width_array = []
        heights_array = []
    
        rounded_prominence = []
    
        for element in (properties["prominences"]):
            rounded_prominence.append(round(element, 4))
    
        prominent_peaks = properties["prominences"][0:int(len(properties["prominences"])/2)]
        sorted_peaks = np.asarray(sorted(prominent_peaks, reverse = True))
    
        index_params = []
    
        for element in sorted_peaks:
            index_element = np.where(rounded_prominence == (round(element, 4)))[0]
            #print(index_element)
            
            
            quadrant_value = False
            for j in range(len(index_element)):
                if abs(x_peak_array[peaks][index_element[j]]) > 3.0 and abs(y_peak_array[peaks][index_element[j]]) > 0:
                    quadrant_value = True
            
    
            if len(index_element) == 2 and quadrant_value == True:
                lattice_parameter = 2/(distance(x_peak_array[peaks][index_element],y_peak_array[peaks][index_element]))
    
                if lattice_parameter <= 0.3 and lattice_parameter >= 0.15:
    
                    index_params.append(index_element)
    
                    lattice_param_array.append(lattice_parameter)
    
                    prominence_array.append(properties["prominences"][index_element[0]])
                    width_array.append(properties["widths"][index_element[0]])
                    heights_array.append((fft_data[i])[peaks][index_element[0]])
    
        print(f"shortlisted points for {i+1} scan = {len(width_array)}")        
        
        '''
        
        peak_data = hexagon_peak_find(fft_data[i], pixels, frame_width, norm_factor, xdata, ydata)
        
        
        peak_height_array = peak_data[1]
        
        
        peak_variance = fft_peak_variance(file_name, 5, pixels)[2]
        
        
        
        
        
        if len(peak_height_array) > 0:
    
            abs_avg_height = peak_data[4]
            norm_avg_height = peak_data[3]
            
            
        else:
    
            abs_avg_height = 0
            norm_avg_height = 0
    
    
    
    
    else:

        abs_avg_height = -2        
        peak_variance = -1
        norm_avg_height = -1

    parameter_array = []
    parameter_array_r = []
    
    parameter_array.append(round(abs_avg_height, 3))    
    parameter_array.append(round(norm_avg_height, 3))
    parameter_array.append(round(len(peak_height_array), 3))
    parameter_array.append(round(peak_variance, 6))
    
    parameter_array = np.asarray(parameter_array)
    
    for element in parameter_array:
        parameter_array_r.append(element)
        
    return parameter_array_r
    
    
