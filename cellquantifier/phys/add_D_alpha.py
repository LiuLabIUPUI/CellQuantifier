import numpy as np
import pandas as pd
import trackpy as tp
from ..qmath import fit_msd1, fit_msd1_log, fit_msd2

def add_D_alpha(df, pixel_size, frame_rate, divide_num,
	fit_method='fit_msd1_log'):

	for col in ['D', 'alpha']:
		if col in df:
			df = df.drop(col, axis=1)

	df_cut = df[['frame', 'x', 'y', 'particle']]
	im = tp.imsd(df_cut, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)

	n = int(round(len(im.index)/divide_num))
	im = im.head(n)

	#get diffusion coefficient of each particle
	particles = im.columns
	ind = 1
	tot = len(particles)
	for particle in particles:
		print("(%d/%d)" % (ind, tot))
		ind = ind + 1

		# Remove NaN, Remove non-positive value before calculate log()
		msd = im[particle].dropna()
		msd = msd[msd > 0]

		if len(msd) > 2: # Only fit when msd has more than 2 data points
			x = msd.index.values
			y = msd.to_numpy()
			y = y*1e6 #convert to nm

			if fit_method=='fit_msd2':
			    popt = fit_msd2(x, y)
			elif fit_method=='fit_msd1':
			    popt = fit_msd1(x, y)
			else:
			    popt = fit_msd1_log(x, y)

			df.loc[df['particle']==particle, 'D'] = popt[0]
			df.loc[df['particle']==particle, 'alpha'] = popt[1]

	return df
