#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:00:00 2021

@author: dbenet
"""

# 4Oct2022: Added qia.py functions

import re, cv2, glob, shutil, os, math, csv, pylab, PIL
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from operator import itemgetter

import skimage
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.draw import rectangle,polygon,ellipse_perimeter
from skimage import img_as_float
from skimage.util import img_as_ubyte

import pandas as pd
import seaborn as sns

from scipy import stats

from mpl_toolkits.mplot3d import Axes3D
from mycolorpy import colorlist as mcp
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as lines

from PIL import Image
from imblearn.over_sampling import SMOTE

##### PREPARE DF ####
# Add basic volcano info to dataframe
def add_basic_volcano_info(df, multifocus: bool):
	df.dropna(inplace=True)
	df['luminance'] = (df['red_mode']*0.22) + (df['green_mode']*0.71) + (df['blue_mode']*0.07)
	if multifocus == True:
		particle_types = []
		for i in df.index:
			s = str(i)
			s = s.split('_')[-1]
			particle_types.append(s)
		df['Particle type'] = particle_types
	else: pass
	df['Grain-size'] = [label[-2] for label in df.index.str.split('_')]
	df['Volc'] = [idx.split('-')[0] for idx, row in df.iterrows()] 
	df['Volcano'] = df.Volc.replace({'CV':'Cumbre Vieja', 'PI':'Pinatubo', 'NC':'Nevados de Chillán',
	 'ME':'Merapi', 'MS':'Mount St Helens', 'ON':'Ontake', 'SG':'Soufrière de Guadaloupe', 'TO':'Toba', 'KE':'Kelud'})
	df['Eruptions'] = [idx.split('_')[0] for idx, row in df.iterrows()]
	df['Aliquote'] = ['_'.join(idx.split('_')[:2]) for idx, row in df.iterrows()]
	if multifocus == True:
		df['Technique'] = 'Multifocus'
	else: df['Technique'] = 'Common'
	style_list = []
	for idx, value in df['Eruptions'].items():
		if value == 'KE-DB2' or value =='KE-DB3':
			style = 'Subplinian' 
		elif value in ['ON-DB1','PI-DB1','SG-DB1','SG-DB2']:
			style = 'Phreatic' 
		elif value in ['NC-DB5', 'NC-DB4', 'NC-DB2', 'NC-DB6', 'NC-DB1', 'PI-DB3', 'NC-DB3', 'ME-DB1', 'ME-DB2']:
			style = 'Dome explosion' 
		elif value in ['NC-DB15', 'NC-DB7', 'NC-DB8', 'NC-DB9']:
			style = 'Dome explosion' # maybe could be argued that it is a bit more Strombolian?       
		elif value == 'CV-DB1':
			style = 'Lava fountaining'
		elif value in ['MS-DB1','TO-DB1', 'PI-DB3']:
			style = 'Plinian'
		elif value[:2] == 'TV':
			style = 'Phreatic historical'
		else: style = 'NaN'
		style_list.append(style)
	df['Eruptive style'] = style_list
	volcano_type = []
	for idx, value in df['Eruptions'].items():
		if value[:2] in ['KE','ME','ON','SG']:
			volcano_t = 'Stratovolcano' 
		elif value[:2] in ['PI','TO']:
			volcano_t = 'Caldera'
		elif value[:2] == 'NC':
			volcano_t = 'Dome complex'       
		elif value[:2] == 'CV':
			volcano_t = 'Cinder cone'
		else: volcano_t = 'Stratovolcano'
		volcano_type.append(volcano_t)
	df['Volcano type'] = volcano_type
	magma_composition = []
	for idx, value in df['Eruptions'].items():
		if value[:2] in ['KE','ME','ON','SG','NC']:
			composition = 'Intermediate' 
		elif value[:2] in ['MS','TO','PI']:
			composition = 'Silicic'       
		elif value[:2] == 'CV':
			composition = 'Mafic'
		else: composition = 'NaN'
		magma_composition.append(composition)
	df['Magma composition'] = magma_composition

	basic_component = []
	for i in df['Particle type']:
		if i[:2] == 'PG' or i[:2] == 'PX' or i[:3] == 'AMF' or i[:2] == 'SU' or i[:2] == 'OL':
			component = 'Free-crystal'
		elif i[:2] == 'AW' or i[:2] == 'AH':
			component = 'Altered material'
		elif i[:2] == 'LL' or i[:2] == 'LJ':
			component = 'Lithic'
		elif i[:2] == 'JJ' or i[:2] == 'JH':
			component = 'Juvenile'
		else:
			component = np.nan
		basic_component.append(component)
	df['Main type'] = basic_component

	subtypes = []
	for i in df['Particle type']:
		if i in ['PG','PX','AMF','SU','OL']:
			component = i
		elif i[:2] == 'AW':
			component = 'Weathered material'
		elif i[:2] == 'AH':
			component = 'Hydrothermally altered material'
		elif i[:2] == 'LL':
			component = 'Standard lithic'
		elif i[:2] == 'LJ':
			component = 'Recycled juvenile'	
		elif i[:2] == 'JJ':
			component = 'Standard juvenile'
		elif i[:2] == 'JH' or i[:2] == 'JH':
			component = 'Syneruptively altered juvenile'
		else:
			component = np.nan
		subtypes.append(component)
	df['Sub-type'] = subtypes

	subtypes = []
	for i in df['Particle type']:
		if i in ['PG','PX','AMF','SU','OL']:
			component = i
		elif i[:2] == 'AW':
			component = 'AW'
		elif i[:2] == 'AH':
			component = 'AH'
		elif i[:2] == 'LL':
			component = 'LL'
		elif i[:2] == 'LJ':
			component = 'LJ'	
		elif i[:2] == 'JJ':
			component = 'JJ'
		elif i[:2] == 'JH' or i[:2] == 'JH':
			component = 'JH'
		else:
			component = np.nan
		subtypes.append(component)
	df['Sub-type_abbr'] = subtypes

	crystallinity = []
	for i in df['Particle type']:
		if 'trlc' in i:
			component = 'trlc'
		elif 'bllc' in i:
			component = 'bllc'
		elif 'trmc' in i:
			component = 'trmc'	
		elif 'blmc' in i:
			component = 'blmc'
		elif 'trhc' in i:
			component = 'trhc'	
		elif 'blhc' in i:
			component = 'blhc'
		else:
			component = np.nan
		crystallinity.append(component)
	df['Crystallinity'] = crystallinity

	hydrothermal_degree = []
	for i in df['Particle type']:
		if i[:2] == 'JH':
			if i[6] == 'h':
				component = 'high'
			elif i[6] == 'm':
				component = 'medium'
			elif i[6] == 'l':
				component = 'low'	
		else:
			if i[-1] == 'h':
				component = 'high'
			elif i[-1] == 'm':
				component = 'medium'
			elif i[-1] == 'l':
				if i[-2:] != 'ol':
					component = 'low'	
			elif i[-1] == 'n':
				component = 'absent'
			else:
				component = np.nan
		hydrothermal_degree.append(component)
	df['Hydrothermal degree'] = hydrothermal_degree

	vesicularity = []
	for i in df['Particle type']:
		if i[-1] == 'b' or i[-1] == 'f':
				component = 'non-vesicular'
		elif i[-1] == 's' or i[-2:] == 'hv' or i[-2:] == 'mt' or i[-1] == 'p':
			component = 'vesicular'	
		else:
			component = np.nan
		vesicularity.append(component)
	df['Vesicularity'] = vesicularity

	shape = []
	for i in df['Particle type']:
		if i[-1] == 'b':
			component = 'blocky'
		elif i[-1] == 'f':			
			component = 'fluidal'
		elif i[-1] == 's':
			component = 'spongy'	
		elif i[-2:] == 'hv':
			component = 'highly-vesicular'
		elif i[-2:] == 'mt':
			component = 'microtubular'	
		elif i[-1] == 'p':
			component = 'pumice'
		else:
			component = np.nan
		shape.append(component)
	df['Shape'] = shape
	return df


def sorted_alphanumeric(data):
	'''Natural sorting using re'''
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 

	return sorted(data, key=alphanum_key)

def scale_df(X_train, X_test):
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	return X_train_scaled, X_test_scaled

def scale_df_and_output_df(X_train, X_test, columns):
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	join = np.concatenate((X_train_scaled, X_test_scaled), axis = 0)
	df = pd.DataFrame(join, columns = columns, index = mf.index)

	return df

#### QIA ####
def mask(ch4):

	alpha_ch = ch4[...,3]
	rgb_image = ch4[...,:3]
	global thr
	ret, thr = cv2.threshold(alpha_ch, 120, 255, cv2.THRESH_BINARY)
	global height
	global width
	height, width = thr.shape

	contours, hier = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	global contour
	contour = max(contours, key = len)
	part_mask = np.zeros(rgb_image.shape)
	part_mask = cv2.drawContours(part_mask, [contour], -1,(255,255,255), -1)
	rgb_masked = rgb_image.copy()
	rgb_masked[np.logical_not(part_mask)] = 0      
	part = 255*np.ones(alpha_ch.shape) 
	part = cv2.drawContours(part, [contour], -1, (0), 5)
	#cv2.imwrite('part.png', part)

	return part_mask[...,0], rgb_masked

def shape(alpha_ch):
	# find contour and image moments
	M = cv2.moments(contour)
	minRect = cv2.minAreaRect(contour)
	_, (w, l), _ = minRect		 
	# skimage function to select a roi
	label_image = label(alpha_ch)
	global regions
	regions = regionprops(label_image)
	print(f'Detected regions: {len(regions)}')
	area = [ele.area for ele in regions] 
	largest_region_idx = np.argmax(area)
	global props
	props = regions[largest_region_idx]   
	major_ellipse_axis = props.major_axis_length
	minor_ellipse_axis = props.minor_axis_length
	# particle descriptives
	global y_centroid
	global x_centroid
	y_centroid, x_centroid = props.centroid
	part_perim = props.perimeter    
	area = props.area
	# calculate hull
	hull = convex_hull_image(alpha_ch)
	hull_perim = measure.perimeter(hull)
 

	
	# measure properties
	eccentricity_moments = eccentricity_from_moments(M)
	eccentricity_ellipse = eccentricity_from_ellipse(contour)
	eccentricity = props.eccentricity
	aspect_rat = aspect_ratio(contour) # this is imported from helper.py and uses width and height, which is not as precise as Dminferret and Dmaxferret in Liu
	solidity = props.solidity
	convexity = hull_perim / part_perim #  usually the value is not greater than 1
	circularity_dellino = part_perim/(2*math.sqrt(math.pi*area))
	circ_func = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)
	circ = circ_func(props)
	#print(f'Circularity: Dellino {circularity_dellino}\nCioni{circ}')
	rectangularity = part_perim / (2*l + 2*w)
	compactness = area / (l * w)
	elongation = (props.feret_diameter_max ** 2) / area
	roundness = 4*area / (math.pi * (props.feret_diameter_max ** 2))
	# important factors
	circ_rect = circularity_dellino * rectangularity
	comp_elon = compactness * elongation
	circ_elon = circularity_dellino * elongation
	rect_comp = rectangularity * compactness

	# # for plot purposes
	# black = 255*np.ones(rgb_masked.shape, dtype=np.uint8)
	# black = cv2.cvtColor(black, cv2.COLOR_RGB2RGBA)
	# black[:, :, 3] = alpha_ch
	# cv2.fillPoly(black, pts =[contour], color=(200,200,200, 255))
	# hull = convex_hull_image(alpha_ch)
	# hull_np = np.array(hull, dtype=np.uint8)
	# cnts_hull, hier = cv2.findContours(hull_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# contour_hull = cnts_hull[0]
	# # hull_perim = measure.perimeter(hull)
	# # hull_diff = img_as_float(hull.copy())
	# # hull_diff[alpha_ch] = 2 # color intersection in gray

	# #mask_3d = black[:, :, None] * np.ones(3, dtype=int)*255[None, None, :]
	# cv2.fillPoly(black, pts =[contour_hull], color=(40,180,40, 80))
	# cv2.drawContours(black, contour_hull,-1,[40,40,180, 255],5)
	# cv2.drawContours(black, contour,-1,[0,0,0, 255],5)
	# plt.imshow(black)
	# plt.title(f'{i}_{convexity}')
	# plt.show()
	# plt.close()
 
	shape_dict = {'convexity':convexity,'rectangularity':rectangularity, 'elongation':elongation, 'roundness':roundness, 'circularity':circularity_dellino, 'eccentricity_moments':eccentricity_moments,'eccentricity_ellipse':eccentricity_ellipse,'solidity':solidity, 'aspect_rat':aspect_rat, 'compactness':compactness, 'circ_rect':circ_rect, 'comp_elon':comp_elon, 'circ_elon':circ_elon, 'rect_comp':rect_comp}
	return shape_dict

def texture(image):
	gray = rgb2gray(image)
	image = img_as_ubyte(gray) 
	bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
	inds = np.digitize(image, bins) # makes bins for the pixel range from 0 to 15   


	max_value = inds.max()+1

	# PATCH_SIZE = int(props.major_axis_length/50) # this has to be large enough for small images
	# STEP = int(props.major_axis_length/20)
	thetas = [0,  np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, np.pi+np.pi/4, np.pi+np.pi/2, np.pi+3*np.pi/4, 
	np.pi/4 + np.pi/8, np.pi/2 + np.pi/8, 3*np.pi/4 + np.pi/8, np.pi + np.pi/8, np.pi+np.pi/4 + np.pi/8, 
	np.pi+np.pi/2 + np.pi/8, np.pi+3*np.pi/4 + np.pi/8]

	# # define locations within the particle at a given step
	# locs = []

	# for theta in thetas:
	# 	for ii in range(1000):
	# 		i = ii+1
	# 		new_y = int(y_centroid-np.sin(theta)*STEP*i)
	# 		new_x = int(x_centroid+np.cos(theta)*STEP*i)
	# 		if new_x < width and new_x > 0 and new_y < height and new_y > 0:
	# 			co = (new_y, new_x)
	# 			if alpha_ch[co]>0:
	# 				locs.append(co)
	# 		else: break

	# # take patches that are fully within the particle at a PATCH_SIZE

	# locs_no_background = []
	# patches = []

	# for idx, loc in enumerate(locs):
	# 		patch = (alpha_ch[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
	# 		if patch.min() > 0:
	# 			glcm_patch = (inds[loc[0]:loc[0] + PATCH_SIZE,loc[1]:loc[1] + PATCH_SIZE])
	# 			locs_no_background.append((loc[0], loc[1]))
	# 			patches.append(glcm_patch)

	# # plt.imshow(thr)
	# # ys, xs = zip(*locs)
	# # plt.plot(xs, ys, 'ro')
	# # plt.title(filename)
	# # plt.show()
	# # plt.close()
	# # define distances at which the target pixel will be compared to the origin pixel for the glcm
	# distances = [1]

	# for i in range(5):
	# 	d = int((i+1)/5 * PATCH_SIZE)
	# 	distances.append(d)

	# # init lists for texture features
	# contrast_patches = []
	# dissimilarity_patches = []
	# homogeneity_patches = []
	# energy_patches = []
	# correlation_patches = []
	# asm_patches = []

	#for idx, patch in enumerate(patches):
		# this checks whether at an angle theta the target pixel is not out of the particle
		#print(f'Computing GLCM of patch {idx}/{len(patches)} located at {locs[idx]}')
	matrix_coocurrence = graycomatrix(inds, [1,3, 5], thetas, levels=max_value, normed=False, symmetric=False)
	contrast = graycoprops(matrix_coocurrence, 'contrast')
	contrast_mean = contrast.mean()   
	dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')    
	dissimilarity_mean = dissimilarity.mean()   
	homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
	homogeneity_mean = homogeneity.mean()   
	energy = graycoprops(matrix_coocurrence, 'energy')
	energy_mean = energy.mean()
	correlation = graycoprops(matrix_coocurrence, 'correlation')
	correlation_mean = correlation.mean()   
	asm = graycoprops(matrix_coocurrence, 'ASM')
	asm_mean = asm.mean()

	texture_dict = {'contrast':contrast_mean,'dissimilarity':dissimilarity_mean,'homogeneity':homogeneity_mean,'energy':energy_mean,'correlation':correlation_mean,'asm':asm_mean}
	print(len(texture_dict))
	return texture_dict

def color(image):
	color_dict = {}
	for i, c in enumerate(['blue','green','red']):
		channel = image[...,i]
		values = channel[channel>0] # this is to remove the background
		color_dict[f'{c}_mean'] = values.mean() 
		color_dict[f'{c}_std'] = values.std()
		color_dict[f'{c}_mode'] = int(stats.mode(values)[0]) 
		color_dict[f'{c}_skew'] = int(stats.skew(values))

		hist,_ = np.histogram(values, bins=255)
		hist = zeros2midpoints(hist)
		
		slopes = []
		for i, v in enumerate(hist):
			if i > 0:
				curr = hist[i]
				prev = hist[i-1]
				slope = abs(curr-prev)
				slopes.append(slope)

		no_pixels = len(values)
		#color_dict[f'{c}_slope'] = ((sum(slopes)/len(slopes))/no_pixels)*100

	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
	for i, c in enumerate(['hue','saturation','value']):
		channel = hsv_image[...,i]
		values = channel[channel>0] # this is to remove the background. Black has a hue angle of 0 degrees, a saturation of 0% and a lightness of 0%.
		color_dict[f'{c}_mean'] = values.mean() 
		color_dict[f'{c}_std'] = values.std()
		color_dict[f'{c}_mode'] = int(stats.mode(values)[0]) 

	return color_dict

#### ERRORS ####
# def calculate_margin_of_error_sample(pd_series):
# 	n = pd_series.sum()
# 	proportions= pd_series/n
# 	moe_list = []
# 	for proportion in proportions:
# 		prop_part = [1] * int((proportion * n))
# 		prop_others = [0] * int(((1-proportion) * n))
# 		sample = prop_part + prop_others
# 		mean = np.mean(sample)
# 		SEM = np.std(sample)/(math.sqrt(len(sample)))
# 		confidence_interval = 1.96 # second std
# 		MOE = confidence_interval * SEM*100
# 		moe_list.append(MOE)
# 	return moe_list

def calculate_margin_of_error_sample(pd_series):
	'''
	Calculates the margin of error for each component 
	given a number of particles in a pandas series

	'''
	n = pd_series.sum()
	proportions = pd_series/pd_series.sum()
	moe_list = [] # absolut error
	rel_errors = []
	for proportion in proportions:
		particles = proportion * n
		others = (1-proportion) * n
		mean = (others * 0 + particles * 1)/n # the mean is indeed the proportion
		variance = (others*(0-proportion)**2 + (particles*(1-proportion)**2))/(n-1)
		standard_deviation = variance ** 0.5
		SEM = standard_deviation/(n ** 0.5)
		confidence_interval = 1.96 # second std
		MOE = confidence_interval * SEM * 100 # where it represents the absolut error
		moe_list.append(MOE)
		if proportion > 0:
			rel_error = (MOE/proportion)*100
		else: rel_error = 0
		rel_errors.append(rel_error)
	return moe_list#, rel_errors

def calculate_MOE_from_proportions(proportions, n):
	'''
	Calculates the margin of error for each component 
	given the proportions and sample size

	'''
	moe_list = []
	for proportion in proportions:
		particles = proportion * n
		others = (1-proportion) * n
		mean = (others * 0 + particles * 1)/n # the mean is indeed the proportion
		variance = (others*(0-proportion)**2 + (particles*(1-proportion)**2))/(n-1)
		standard_deviation = variance ** 0.5
		SEM = standard_deviation/(n ** 0.5)
		confidence_interval = 1.96 # second std
		MOE = confidence_interval * SEM * 100 # where it represents the absolut error
		moe_list.append(MOE)
	return moe_list

def calculate_MOE_from_one_proportion(proportion, n, z_score):
	'''
	Calculates the margin of error of one component 
	given its proportion and sample size

	'''
	if proportion > 0:
		particles = proportion * n
		others = (1-proportion) * n
		mean = (others * 0 + particles * 1)/n # the mean is indeed the proportion
		variance = (others*(0-proportion)**2 + (particles*(1-proportion)**2))/(n-1)
		standard_deviation = variance ** 0.5
		SEM = standard_deviation/(n ** 0.5)
		MOE = z_score * SEM # where it represents the absolut error
	else: MOE = 0
	return MOE

def MOE_calculator(proportion, n):
	'''
	Calculates the margin of error of one component 
	given its proportion and sample size

	'''
	if proportion > 0:
		particles = proportion * n
		others = (1-proportion) * n
		mean = (others * 0 + particles * 1)/n # the mean is assumed the real population proportion
		variance = (others*(0-proportion)**2 + (particles*(1-proportion)**2))/(n-1)
		standard_deviation = variance ** 0.5
		SEM = standard_deviation/(n ** 0.5)
		confidence_interval = 1.96 # second std
		MOE = confidence_interval * SEM * 100 # where it represents the absolut error
		print(f'Absolut error = {MOE}\nRelative error = {MOE/proportion*100}')
	else: MOE = 0
	return MOE

#def calculate_margin_of_error_proportion():

#### PLOTS ####
# Simple piecharts
def frequency_piechart_1col(df, col1):
	fig, ax = plt.subplots()
	data = df.groupby(col1).size()
	data.plot.pie(autopct = '%.1f%%', title = col1)
	#plt.title(col1)
	pylab.ylabel('')
	if df['Technique'][0] == 'Multifocus':	
		plt.savefig(f'./results/pie_mf_{col1}.svg', dpi = 400)
	else: plt.savefig(f'./results/pie_bi_{col1}.png', dpi = 400)
	plt.close()

def my_fmt(x):
	return '{:.1f}%\n({:.0f})'.format(x, total*x/100)

def frequency_piechart_cols(df, cols):
	subset = df.loc[:,cols]
	fig, axes = plt.subplots(1,3, figsize = (24,8))
	for ax, col in zip(axes.flatten(), subset):
		data = df.groupby(col).size()
		global total
		total = data.sum()
		data.plot.pie(autopct = my_fmt, title = col, ax = ax)
		ax.set_ylabel('')
	col_names = '_'.join(cols)
	if df['Technique'][0] == 'Multifocus':
		fig.suptitle('Multifocus dataset', fontsize = 16, fontweight = 'bold')
		plt.savefig(f'./results/pie_mf_{col_names}.png', dpi = 400)
	else: 
		fig.suptitle('Binocular dataset', fontsize = 16, fontweight = 'bold')
		plt.savefig(f'./results/pie_bi_{col_names}.png', dpi = 400)
	plt.close()

# Piechart basic info
def piechart_compo(df, col1):

	no_groups = len(df.groupby(col1).size())
	types = list(df.groupby(col1).size().index)

	frequency = list(df.groupby(col1).size().values)
	zip_iterator = zip(types, frequency)
	dict_color = dict(zip_iterator)

	wong_palette = [
	#"#000000",
	"#E69F00",
	"#56B4E9",
	"#009E73",
	"#F0E442",
	"#0072B2",
	"#D55E00",
	"#CC79A7",
	]


	labels = []
	#colors = []

	for idx, (key,value) in enumerate(dict_color.items()):
		#color = [wong_palette[idx+1]] # to avoid black
		x = [key]
		x = x*value
		#color = color*value
		labels.append(x)
		#colors.append(color)

	labels = [item for sublist in labels for item in sublist] # flatten list
	#colors = [item for sublist in colors for item in sublist] # flatten list

	labels = df[col1]
	le = preprocessing.LabelEncoder()
	le.fit(labels)

	iidxs = le.transform(df[col1])
	colors = [wong_palette[i] for i in iidxs]

	global total
	total = df['Total'].sum()
	size = 1
	fontsize_legend = 12

	fig, ax = plt.subplots(figsize=(9,9))
	
	if 'Eruptions' in df:
		ax.pie(
			df['%'], radius=1, colors = colors, autopct = my_fmt,#autopct = '%.1f%%',
			labels = df['Eruptions'], startangle=0,wedgeprops=dict(width=size, edgecolor='black')
			)

	else:
		ax.pie(
			df['%'], radius=1, colors = colors, autopct = my_fmt,#autopct = '%.1f%%',
			labels = df['Volcano'], startangle=0,wedgeprops=dict(width=size, edgecolor='black')
			)

	seen = set()
	no_dup_labels = []
	for item in labels:
		if item not in seen:
			seen.add(item)
			no_dup_labels.append(item)

	seen = set()
	no_dup_colors = []
	for item in colors:
		if item not in seen:
			seen.add(item)
			no_dup_colors.append(item)

	# print(labels)
	# uniq, indexes = np.unique(labels2, return_index=True)
	# print(uniq)
	# #print(len(indexes))
	# l = [None] * len(indexes)
	# for index, label in zip(indexes,uniq):
	# 	print(index)
	# 	l[index] = label

	legend_elements = []
	for color, label in zip(no_dup_colors,no_dup_labels):
		element = mpatches.Patch(color = color, label=label, edgecolor = 'k')
		element.set_edgecolor('black')
		element.set_linewidth(0.2)
		legend_elements.append(element)

	legen = plt.legend(frameon = False, handles = legend_elements, bbox_to_anchor=(0.9,1.01), title = col1)#,, fontsize = fontsize_legend bbox_to_anchor=(-4.375, 0.9))
	plt.tight_layout()

	if 'Eruptions' in df:
		plt.savefig(f'./results/samples_{col1}_volcano_compo.png', dpi = 400)
		plt.show()
		plt.close()

	else:
		plt.savefig(f'./results/volcanoes_{col1}_volcano_compo.png', dpi = 400)
		plt.show()
		plt.close()

def plot_3d_plt (df, red_col, green_col, blue_col, y_col):
	#plt.style.use('dark_background')
	colors = ['orange','lightblue','darkblue','red','grey','purple','darkgreen']
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	for idx, class_part in enumerate(df[y_col].unique()):
		
		df2 = df[df[y_col] == class_part]
		ax.scatter(df2[red_col], df2[green_col],df2[blue_col], marker='o', s=20, c = colors[idx], alpha = 0.5, label = class_part)
	
	ax.legend(fontsize = 8)
	ax.set_xlabel('Red channel')
	ax.set_ylabel('Green channel')
	ax.set_zlabel('Blue channel')
	
	ax.legend(loc = (0.1, 0.6), fontsize = 8)
	fig.set_facecolor('black')
	ax.set_facecolor('black') 
	
	plt.tight_layout()
	plt.savefig('./figures/3d_scatter.png', dpi=600)

def plot_multiple_histograms(df, cols):
  num_plots = len(cols)
  num_cols = math.ceil(np.sqrt(num_plots))
  num_rows = math.ceil(num_plots/num_cols)
        
  fig, axs = plt.subplots(num_rows, num_cols, figsize = (20,15))
    
  for ind, col in enumerate(cols):
    i = math.floor(ind/num_cols)
    j = ind - i*num_cols
            
    if num_rows == 1:
      if num_cols == 1:
        sns.distplot(df[col], bins=256, kde=True, ax=axs, color="black")
      else:
        sns.distplot(df[col], bins=256, kde=True, ax=axs[j], color="black")
    else:
        sns.distplot(df[col], bins=256, kde=True, ax=axs[i, j], color="black")
    
  for ax in axs.flatten():
    ax.set_ylabel('')

  plt.tight_layout()
  #plt.savefig(f'./dataset_process_results/{keyword_to_save}.svg', dpi = 400)
		
# Model metrics
def plot_cm(y_true, y_pred, labels, figsize=(14, 11), colour_by_perc=False, cmap='Greens'):
	cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
	cm_sum = np.sum(cm, axis=1, keepdims=True)
	cm_perc = cm / cm_sum.astype(float) * 100
	annot = np.empty_like(cm).astype(str)
	nrows, ncols = cm.shape
	for i in range(nrows):
		for j in range(ncols):
			c = cm[i, j]
			p = cm_perc[i, j]
			if i == j:
				s = cm_sum[i]
				annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
			elif c == 0:
				annot[i, j] = ''
			else:
				annot[i, j] = '%.1f%%\n%d' % (p, c)
	if colour_by_perc:
		cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
	else:
		cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
	cm.index.name = 'Database types'
	cm.columns.name = 'Predicted types'
	fig, ax = plt.subplots(figsize=figsize)
	g = sns.heatmap(cm, annot=annot, fmt='', ax=ax, square=True, xticklabels=labels, yticklabels=labels, cmap = cmap, annot_kws={"size": 16})
	g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 16)
	g.set_yticklabels(g.get_xmajorticklabels(), fontsize = 16)
	plt.xticks(rotation=45)
	plt.title('Confusion Matrix')
	plt.tight_layout()

def plot_bar_cm(y_labels, y_pred, dict_target_inv, main_type:bool):
	
	if main_type:
		colors_dic = {'Altered material':'#F7931E','Free-crystal':'#29ABE2','Juvenile':'#B81A5D','Lithic':'#006837'}
		colors_final = {}

		for k,v in dict_target_inv.items():
			color = colors_dic[v]
			colors_final[v] = color

	else:
		colors_final = ['lightblue','darkblue']

	viz_df = pd.DataFrame({'Main type':y_labels,'Predicted':np.vectorize(dict_target_inv.get)(y_pred)})
	viz_df = viz_df.groupby('Predicted')['Main type'].value_counts().unstack(fill_value=0)
	scaled = pd.DataFrame(index = viz_df.index, columns = viz_df.columns)

	# to normalize to 100
	for col in viz_df:
		column = viz_df.loc[:,col]
		scaled[col] = (column.values/column.values.sum())*100

	g = scaled.T.plot(kind='bar', stacked=True, color=colors_final)
	plt.ylabel('Proportion (%)')
	plt.xticks(rotation = 0)
	plt.xlabel('Database types')
	g.figure.set_figheight(4)
	g.figure.set_figwidth(6)
	g.legend(bbox_to_anchor = (0.5,1.05), ncol = 4, fontsize = 8, loc = 'center', frameon=False, title = 'Predicted types')
	plt.tight_layout()
	
	
#### MODELING ####

def prepare_Xs_and_Ys(mf, variable_start, variable_end, target, balance_dataset: bool, outlier: str, rescale_dataset: bool, filter_column: str, classes2keep: list, volcano: str):
	
	if volcano != None:
		mf = mf[mf['Volcano'] == volcano]

	if filter_column != None:
		mf = mf[mf[filter_column].isin(classes2keep)]

	df = mf.loc[:,variable_start:variable_end]
	df = pd.concat([df,mf[target]], axis = 1)
	df = df.dropna()
	if target == 'Eruptive style':
		#order_labels = pd.CategoricalDtype(['ON', 'PI', 'SG', 'TV', 'KE'], ordered=True)
		order_labels = pd.CategoricalDtype(df[target].unique(), ordered=True)
	elif target == 'Textural group':
		order_labels = pd.CategoricalDtype(['Amphibole','Plagioclase',
			'Pyroxene','Altered material','Weathered material',
			'LCTR', 'LCBL', 'MCBL', 'MCTR', 'HCTR','HCBL'], ordered=True)
	else:
		order_labels = pd.CategoricalDtype(df[target].unique(), ordered=True)
	df[target] = df[target].astype(order_labels)
	df = df.sort_values(target)

	# create a new column with Y's factors
	df['Y_num'],_ = pd.factorize(df[target])
	if balance_dataset:
		# x = df.groupby('Basic component').size() > 20
		# l = x[x].index
		# df = df[df['Basic component'].isin(l)]
		less_frequent_class = df.groupby(df[target]).size().min()
		df = df.groupby(df[target]).head(less_frequent_class)

	# detect outliers
	df_zscores = pd.DataFrame()
	for name, col in df.loc[:,variable_start: variable_end].iteritems():
		df_zscores[name] = (col.values - col.values.mean()) / col.values.std()
	df_zscores.index = df.index
	df_outliers = df_zscores[abs(df_zscores) > 3]
	df_outliers.dropna(how='all',inplace=True)
	outlier_names = df_outliers.index

	# process outliers
	if outlier == 'unprocessed':
		pass

	elif outlier == 'deletion':
		df.drop(outlier_names, inplace=True)

	elif outlier == 'median':
		for name, col in df.loc[:,variable_start: variable_end].iteritems():
			med = df[name].quantile(0.50)
			q1 = df[name].quantile(0.25)
			q3 = df[name].quantile(0.75)
			IQR = q3-q1
			lwr_bound = q1-(1.5*IQR)
			upr_bound = q3+(1.5*IQR)

			#df[name] = np.where(df_col2zscore(df[name]) > 3, med, df[name])
			df[name] = np.where((df[name]<lwr_bound) | (df[name]>upr_bound), med, df[name])

	if rescale_dataset:
		Xs = df.loc[:,variable_start:variable_end]
		scaler = MinMaxScaler()
		Xs = scaler.fit_transform(Xs)
		df.loc[:,variable_start:variable_end] = Xs

	return df, df.loc[:,variable_start:variable_end], df['Y_num'], df[target]


# Small piece of code to match labeled images (filenames_img) with the png images and send to dst folder,
#   while keeping the label

def match_and_send_files(folder_img = '/input/images/*/*', folder_png = '/preprocessed2/*'):
	filenames_img = glob.glob(path + folder_img)
	filenames_png = glob.glob(path + folder_png)
	
# Example of dir route
#/Users/dbenet/Desktop/PhD/Chapter2_automatic_class/capytol2/deepL/input/images/altered_material/altered_material_a_30_2_NJDm_350.png
	
	for png_file in filenames_png:
		png_ref = png_file.split('/')[-1].split('.')[0].split('_')
		png_ref = list(itemgetter(0,1,2,4)(png_ref))
		for file_img in filenames_img:
			ref_img_dir = file_img.split('/') # keep this for dstdir
			img_ref = ref_img_dir[-1].split('.')[0].split('_')
			img_ref = list(itemgetter(1,2,3,5)(img_ref))
			if img_ref == png_ref:
				dst_dir = (path + '/input/images2/' + ref_img_dir[-2] + '/' + ref_img_dir[-1].split('.')[0] + '.png')
				shutil.copy(png_file,dst_dir)
				continue
			
			else:
				continue

		
#%% Helper functions for contour analysis
	
def aspect_ratio(contour):
	"""Returns the aspect ratio of the contour based on the dimensions of the bounding rect"""

	x, y, w, h = cv2.boundingRect(contour)
	res = float(w) / h
	return res

def roundness(contour, moments):
	"""Calculates the roundness of a contour"""

	length = cv2.arcLength(contour, True)
	k = (length * length) / (moments['m00'] * 4 * np.pi)
	return k


def eccentricity_from_ellipse(contour):
	"""Calculates the eccentricity fitting an ellipse from a contour"""

	(x, y), (MA, ma), angle = cv2.fitEllipse(contour)

	a = ma / 2
	b = MA / 2

	ecc = np.sqrt(a ** 2 - b ** 2) / a
	return ecc


def eccentricity_from_moments(moments):
	"""Calculates the eccentricity from the moments of the contour"""

	a1 = (moments['mu20'] + moments['mu02']) / 2
	a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
	ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
	return ecc

def solidity(cnt):
	"""Calculates the solidity of from the contour and hull"""
	
	area = cv2.contourArea(cnt)
	hull = cv2.convexHull(cnt)
	hull_area = cv2.contourArea(hull)
	
	solid = float(area)/hull_area
	return solid, hull

def save_dict_to_csv(dict):
	with open('dict.csv', 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in dict.items():
					writer.writerow([key, value])


# to read it use 
# =============================================================================
# with open('dict.csv') as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = dict(reader)
# =============================================================================

# Cool figure for qia.py 

# #left_image = im_gray3d[:,0:int(x_centroid)]
# #right_image = im[:,int(x_centroid):w]
# #merged_im = np.hstack((left_image,right_image))
# up_image = im_gray3d[0:int(y_centroid),:]
# down_image = im[int(y_centroid):h,:]
# merged_im = np.vstack((up_image,down_image))
# cv2.drawContours(merged_im,cnt,-1,[40,40,185],5)
# d = x_centroid-w
# shapestr = '\n'.join((
# r"$\bf{Shape}$", 
# f'Convexity = {convexity:.2f}',
# f'Rectangularity = {rectangularity:.2f}',
# f'Elongation = {elongation:.2f}',
# f'Roundness = {roundness:.2f}',
# f'Circularity = {circularity:.2f}'))

# textstr = '\n'.join((
# r"$\bf{Texture}$", 
# f'Dissimilarity = {dissimilarity_mean:.2f}',
# f'Homogeneity = {homogeneity_mean:.2f}',
# f'Energy = {energy_mean:.2f}',
# f'ASM = {asm_mean:.2f}',
# f'Correlation = {correlation_mean:.2f}'))

# colorstr = '\n'.join((
# r"$\bf{Color}$", 
# f'Red mode = {red_TP_mode:.2f}',
# f'Green mode = {green_TP_mode:.2f}',
# f'Blue mode = {blue_TP_mode:.2f}'))

# fig, ax = plt.subplots(1,2, gridspec_kw={
#      'width_ratios': [0.2, 2]})
#      #'height_ratios': [5,5,1.5,6,5,5]})
# ax[0].text(0.3, 0.75, shapestr, fontsize = 10, va = 'center')
# ax[0].text(0.3, 0.45, textstr, fontsize = 10, va = 'center') 
# ax[0].text(0.3, 0.20, colorstr, fontsize = 10, va = 'center')
# ax[0].axis('off')
# ax[1].imshow(merged_im[...,::-1],)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(f'/Users/dbenet/Desktop/PhD/Chapter2_automatic_class/capytol2/feature_extraction/catalogue2/{complete_name}', transparent = False)
# plt.show()

# else:
# pass

# This is only useful for datasets of two features (sub_X)
def visualize_classifier(model, X, y, title, output_file, ax=None, cmap='rainbow'):
	ax = ax or plt.gca()
	# Plot the training points
	ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap, clim=(y.min(), y.max()), zorder=3, alpha = 0.5)
	ax.axis('tight')
	ax.axis('off')
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	# fit the estimator
	model.fit(X, y)
	xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
						 np.linspace(*ylim, num=200))
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	# Create a color plot with the results
	n_classes = len(np.unique(y))
	contours = ax.contourf(xx, yy, Z, alpha=0.5,
						   levels=np.arange(n_classes + 1) - 0.5,
						   cmap=cmap,
						   zorder=1)
	ax.set(xlim=xlim, ylim=ylim)
	ax.set_title(title)
	plt.savefig('figures/%s.png' % output_file, transparent = True, dpi = 600)
	plt.show()

# return measurements

def resize_image(im, proportion):	
	width, height = im.size
	new_width = int(width/proportion)
	new_height = int(height/proportion)

	new_size = (new_width, new_height)
	resized_im = im.resize(new_size)

	return resized_im

def resize_image_and_save(image_filename, proportion: int):
	PIL.Image.MAX_IMAGE_PIXELS = None
	im = Image.open(image_filename)
	imName = image_filename.split('/')[-1]
	width, height = im.size
	new_width = int(width/proportion)
	new_height = int(height/proportion)

	new_size = (new_width, new_height)
	resized_im = im.resize(new_size)

	resized_im.save(f'resized_{proportion}_{imName}')

def zeros2midpoints(nparray):
	'''Useful to remove zeros after color normalization'''
	idxs = np.where(nparray == 0)[0]
	for i in idxs:
		if i > 0 and i < len(nparray):
			nparray[i] = (nparray[i-1]+nparray[i+1])/2
	return nparray

def visualize_grid(list_of_df_indexes, keyword_to_save):
	no_im = len(list_of_df_indexes)
	if no_im < 25:
		side_grid = int(no_im**0.5)
	else: side_grid = 5
	#fig, axes = plt.subplots(side_grid, side_grid, figsize = (12,10))
	fig, axes = plt.subplots(5, 4, figsize = (12,10))
	for (index, image), ax in zip(enumerate(list_of_df_indexes), axes.flatten()):
		image_filename = f'/Users/dbenet/Desktop/leica_MF/pngs/renamed/{image}.png'
		PIL.Image.MAX_IMAGE_PIXELS = None
		im = Image.open(image_filename)
		proportion = 1
		width, height = im.size
		new_width = int(width/proportion)
		new_height = int(height/proportion)
		new_size = (new_width, new_height)
		resized_im = im.resize(new_size)
		im = np.array(resized_im)
		ax.set_axis_off()
		ax.imshow(im)
		ax.set_title(image.split('_')[-1])
	plt.savefig(f'./results/{keyword_to_save}', dpi=400)

def visualize_rgb(list_of_df_indexes, keyword_to_save):
	no_im = len(list_of_df_indexes)
	if no_im < 25:
		side_grid = int(no_im**0.5)
	else: side_grid = 5
	fig, axes = plt.subplots(side_grid, side_grid, figsize = (12,10))
	for (index, image), ax in zip(enumerate(list_of_df_indexes), axes.flatten()):
		image_filename = f'./pngs/renamed/{image}.png'
		im = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
		colors = ('b','g','r')
		alpha_ch = im[...,3]
		rgb_image = im[...,:3]
		ret, thr = cv2.threshold(alpha_ch, 120, 255, cv2.THRESH_BINARY)
		contours, hier = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		part_mask = np.zeros(rgb_image.shape)
		part_mask = cv2.drawContours(part_mask, contours, -1,(255,255,255), -1)
		rgb_masked = rgb_image.copy()
		rgb_masked[np.logical_not(part_mask)] = 0 
		histograms = []
		for i in range(3):
			hist = cv2.calcHist([rgb_masked], [i], None, [256], [0,255])
			histograms.append(hist)
			ax.plot(hist[1:], color=colors[i])
			#ax.set_xlim(5,256)
			#ax.set_axis_off()
			#ax.imshow(im)
			ax.set_title(image.split('_')[-1])
		if index%5 != 0:
			ax.set_yticks([])
	plt.tight_layout()
	plt.savefig(f'./results/{keyword_to_save}_rgb_grid.png', dpi=400)


def prepare_DL_directories(y_train, y_test, root_dir):

  os.chdir(root_dir)	  
  train_dir = './train/'
  test_dir = './test/'

  di = {0:'altered_material',1:'juvenile',2:'lithic',3:'free_crystal'}
  y_train = y_train.map(di)   
  y_test = y_test.map(di)   

  for i in ([train_dir, test_dir]):
    if not os.path.exists(i):
      os.mkdir(i)
    for ii in y_test.unique().tolist():
      if not os.path.exists(f'{i}{ii}'):
        os.mkdir(f'{i}{ii}')

  y_train.to_csv(f'{train_dir}train.csv')
  y_test.to_csv(f'{test_dir}test.csv')

  images_dir = './pngs/renamed/'

  for i,v in y_train.iteritems():
    src_dir = f'{images_dir}{i}.png'
    dst_dir = f'{train_dir}{v}/{i}.jpg'
    shutil.copy(src_dir, dst_dir)

  for i,v in y_test.iteritems():
    src_dir = f'{images_dir}{i}.png'
    dst_dir = f'{test_dir}{v}/{i}.jpg'
    shutil.copy(src_dir, dst_dir)

### SHAP ###

# tornado plot
def get_ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
 
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    
    k2_f = k2[['Variable', 'SHAP_abs', 'Corr']]
    k2_f['SHAP_abs'] = k2_f['SHAP_abs'] * np.sign(k2_f['Corr'])
    k2_f.drop(columns='Corr', inplace=True)
    k2_f.rename(columns={'SHAP_abs': 'SHAP'}, inplace=True)
    
    return k2_f

# tornado plot
def get_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
 
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    shap_no_abs = shap_v
    k=pd.DataFrame(shap_no_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_no_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_no_abs',ascending = True)
    
    k2_f = k2[['Variable', 'SHAP_no_abs', 'Corr']]
    k2_f['SHAP_no_abs'] = k2_f['SHAP_no_abs'] * np.sign(k2_f['Corr'])
    k2_f.drop(columns='Corr', inplace=True)
    k2_f.rename(columns={'SHAP_no_abs': 'SHAP'}, inplace=True)
    
    return k2_f

def waterfall_plot(index, data, Title="", x_lab="", y_lab="",
              formatting = "{:,.1f}", green_color='#29EA38', red_color='#FB3C62', blue_color='#24CAFF',
             sorted_value = False, threshold=None, other_label='other', net_label='net', 
             rotation_value = 30, blank_color=(0,0,0,0), figsize = (10,10)):
    '''
    Given two sequences ordered appropriately, generate a standard waterfall chart.
    Optionally modify the title, axis labels, number formatting, bar colors, 
    increment sorting, and thresholding. Thresholding groups lower magnitude changes
    into a combined group to display as a single entity on the chart.
    '''
    
    #convert data and index to np.array
    index=np.array(index)
    data=np.array(data)
    
    # wip
    #sorted by absolute value 
    if sorted_value: 
        abs_data = abs(data)
        data_order = np.argsort(abs_data)[::-1]
        data = data[data_order]
        index2 = index[data_order]
    
    #group contributors less than the threshold into 'other' 
    if threshold:
        
        abs_data = abs(data)
        threshold_v = abs_data.max()*threshold
        
        if threshold_v > abs_data.min():
            index = np.append(index[abs_data>=threshold_v],other_label)
            data = np.append(data[abs_data>=threshold_v],sum(data[abs_data<threshold_v]))
    
    changes = {'amount' : data}
    
    #define format formatter
    def money(x, pos):
        'The two args are the value and tick position'
        return formatting.format(x)
    formatter = FuncFormatter(money)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_major_formatter(formatter)

    #Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=changes,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)
    
    trans['positive'] = trans['amount'] > 0

    #Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc[net_label]= total
    blank.loc[net_label] = total

    #The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    #When plotting the last element, we want to show the full bar,
    #Set the blank to 0
    blank.loc[net_label] = 0
    
    #define bar colors for net bar
    trans.loc[trans['positive'] > 1, 'positive'] = 99
    trans.loc[trans['positive'] < 0, 'positive'] = 99
    trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99
    
    trans['color'] = trans['positive']
    
    trans.loc[trans['positive'] == 1, 'color'] = green_color
    trans.loc[trans['positive'] == 0, 'color'] = red_color
    trans.loc[trans['positive'] == 99, 'color'] = blue_color
    
    my_colors = list(trans.color)
    
    #Plot and label
    my_plot = plt.bar(range(0,len(trans.index)), blank, width=0.5, color=blank_color)
    plt.bar(range(0,len(trans.index)), trans.amount, width=0.6,
             bottom=0, color=my_colors)    # !!! change bottom = blank for the waterfall style  
                                   
    
    # connecting lines - figure out later
    #my_plot = lines.Line2D(step.index, step.values, color = "gray")
    #my_plot = lines.Line2D((3,3), (4,4))
    
    #axis labels
    plt.xlabel("\n" + x_lab)
    plt.ylabel(y_lab + "\n")

    #Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)
    
    temp = list(trans.amount)
    
    # create dynamic chart range
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i-1]
    
    trans['temp'] = temp
            
    #plot_max = trans['temp'].max() # !!!uncomment for waterfall
    #plot_min = trans['temp'].min()
    
    plot_max = trans['amount'].max() # !!!comment for waterfall
    plot_min = trans['amount'].min()

    #Make sure the plot doesn't accidentally focus only on the changes in the data
    # if all(i >= 0 for i in temp): # !!!uncomment for waterfall
    #     plot_min = 0
    # if all(i < 0 for i in temp):
    #     plot_max = 0
    
    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)   
    else:
        maxmax = abs(plot_min)
        
    pos_offset = maxmax / 40
    
    plot_offset = maxmax / 15 ## needs to me cumulative sum dynamic

    #Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count

        # if row['amount'] == total:
        #     y = y_height[loop]
        # else:
        #     y = y_height[loop] + row['amount']

        y = row['amount'] # !!! comment this line and uncomment above for proper waterfall

        # Determine if we want a neg or pos offset
        if row['amount'] > 0:
            y += (pos_offset*2)
            plt.annotate(formatting.format(row['amount']),(loop,y),ha="center", color = '#228B22', fontsize=9) # green
        else:
            y -= (pos_offset*4)
            plt.annotate(formatting.format(row['amount']),(loop,y),ha="center", color = '#4E0707', fontsize=9) # red
        loop+=1

    #Scale up the y axis so there is room for the labels
    plt.ylim(plot_min-round(3.6*plot_offset, 7),plot_max+round(3.6*plot_offset, 7))
    
    #Rotate the labels
    plt.xticks(range(0,len(trans)), trans.index, rotation=rotation_value)
    
    #add zero line and title
    plt.axhline(0, color='black', linewidth = 0.6, linestyle="dashed")

    plt.ylabel('Shapley values')
    plt.title(Title)
    plt.tight_layout()

    return fig, ax, index2