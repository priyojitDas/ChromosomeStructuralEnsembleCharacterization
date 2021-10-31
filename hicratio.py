import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import sys

def row(mat):
	rmat = np.ones(mat.shape,dtype='int32')
	rmat = rmat + np.arange(mat.shape[0]).reshape((mat.shape[0],1))
	return rmat

def col(mat):
	cmat = np.ones(mat.shape,dtype='int32')
	cmat = cmat + np.arange(mat.shape[0])
	return cmat

def call_hicratio(mat, bin_size, window_size):

	distance = int(window_size / bin_size)

	rmat = row(mat)

	cmat = col(mat)

	mat[np.abs(rmat-cmat) > distance] = 0

	ins_val = []

	for i in range(mat.shape[0]):
		ileft = np.arange(i-distance+1,i+1,dtype='int32')
		#print ileft.shape
		if np.min(ileft) < 0:
			intra_left = 0
		else:
			mat_left = mat[ileft,:]
			mat_left = mat_left[:,ileft]
			intra_left = np.mean(mat_left[((row(mat_left)-col(mat_left)) >= 0) * (col(mat_left) <= distance)]) 
		
		iright = np.arange(i,i+distance,dtype='int32')
		#print iright.shape
		if np.max(iright) >= mat.shape[0]:
			intra_right = 0
		else:
			mat_right = mat[iright,:]
			mat_right = mat_right[:,iright]
			intra_right = np.mean(mat_right[((row(mat_right)-col(mat_right)) >= 0) * (col(mat_right) <= distance)]) 		

		ibound = np.arange(i-distance+1,i,dtype='int32')
		if np.min(ibound) < 0 or np.max(ibound+distance) >= mat.shape[0]:
			inter = 0
		else:
			mat_inter = mat[ibound,:]
			mat_inter = mat_inter[:,ibound+distance]
			inter = np.mean(mat_inter) * 2

		ins_val.append([intra_left,intra_right,max(intra_left,intra_right),inter])

	ins_val = np.array(ins_val)

	pseudo = np.mean(ins_val[:,3])

	ratio = ins_val[:,2] / (ins_val[:,3] + pseudo)

	peaks, _ = find_peaks(ratio, distance=distance)
	peaks = peaks.tolist()

	tad_boundary = ratio[peaks]
	tad_index = np.arange(mat.shape[0])[peaks]

	tad = pd.DataFrame({'bin':tad_index,'insulation':tad_boundary})

	return (tad, ratio)