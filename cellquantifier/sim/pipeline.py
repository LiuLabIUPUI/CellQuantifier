import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte
from skimage.external.tifffile import imsave
from sinaps.brownian import *
from sinaps.util import *
from sinaps.photon import *


class Movie():

	def __init__(self, settings):

		self.settings = settings
		if self.settings.N_DIM == 2:
			self.settings.Z_SZ = 1
		#ImageJ dimension ordering: TZCXY
		self.shape = (self.settings.NFRAMES,\
					  self.settings.Z_SZ,\
					  self.settings.NCHAN,\
					  self.settings.X_SZ,\
					  self.settings.Y_SZ)
		self.frames = np.zeros(self.shape, dtype=np.float64)

	def build_lang_traj(self, origins=None,
				   corr_mat_3d=None, std=None, method='langevin'):

		"""Wrapper for brownian.build_traj()

		Parameters
		----------
		"""

		self.traj_df = build_traj(nparticles=self.settings.NPARTICLES,
								  nsamples=self.settings.NFRAMES,
								  deg_freedom=self.settings.DEG_FREEDOM,
								  env_temp=self.settings.ENV_TEMP,
								  env_visc=self.settings.ENV_VISC,
								  particle_rad=self.settings.PARTICLE_RADIUS,
								  particle_mass=self.settings.PARTICLE_MASS,
								  dt=self.settings.SAMPLE_PERIOD,
								  damped=True,
								  corr_mat_3d=corr_mat_3d,
								  std=std)

		dx_bar = self.traj_df['dx'].mean()
		lattice_ratio = 25
		self.settings.RESOLUTION = 1000*dx_bar
		lattice_scale = lattice_ratio*self.settings.RESOLUTION #lattice spacing


		#Add in the origins
		if origins is not None:
			origins = origins*lattice_scale
			for n in range(self.settings.NPARTICLES):
				origin = origins[n]
				self.traj_df.loc[self.traj_df['particle'] == n, 'x'] \
								 += float(origin[0])
				self.traj_df.loc[self.traj_df['particle'] == n, 'y'] \
								 += float(origin[1])
				if self.settings.LAT_DIM == 3:
					self.traj_df.loc[self.traj_df['particle'] == n, 'z'] \
									 += float(origin[2])

		self.traj_df['x'] = self.traj_df['x']/self.settings.RESOLUTION
		self.traj_df['y'] = self.traj_df['y']/self.settings.RESOLUTION
		self.traj_df['z'] = self.traj_df['z']/self.settings.RESOLUTION


	def build_brute_traj(self, dcoeff=0, origins=None):

		self.traj_df = build_traj_brute(self.settings.NPARTICLES,
										self.settings.NFRAMES,
										self.settings.FRAME_RATE,
										dcoeff=dcoeff)
		if origins is not None:
			origins = 30*origins
			for n in range(self.settings.NPARTICLES):
				origin = origins[n]
				self.traj_df.loc[self.traj_df['particle'] == n, 'x'] \
								 += float(origin[0])
				self.traj_df.loc[self.traj_df['particle'] == n, 'y'] \
								 += float(origin[1])
				if self.settings.LAT_DIM == 3:
					self.traj_df.loc[self.traj_df['particle'] == n, 'z'] \
									 += float(origin[2])

	def add_photon_stats(self):

		"""Wrapper for dcam.add_photon_stats()

		Parameters
		----------
		"""

		self.traj_df = add_photon_stats(self.traj_df,
										self.settings.EXPOSURE_TIME,
										self.settings.PHOTON_RATE)

	def populate_frame(self, frame, df):

		"""Populates a single frame with particles

		Parameters
		----------

		frame : ndarray
			a single frame to be populated

		df : DataFrame
			DataFrame with x,y,(z) columns
		"""

		nparticles = df['particle'].nunique()
		for i in range(nparticles):
			nphotons = df.loc[df['particle'] == i, 'photons'].to_numpy()[0]
			pos = df.loc[df['particle'] == i, ['x','y','z']]
			pos = pos.to_numpy()[0]
			shape = self.frames.shape
			pos += np.array([shape[3],shape[4],shape[1]])/2
			frame = add_psf(frame,
							nphotons,
							pos,
							ex_wavelen=self.settings.EX_WAVELENGTH,
							em_wavelen=self.settings.EM_WAVELENGTH,
							num_aperture=self.settings.NUM_APERTURE,
							refr_index=self.settings.REFR_INDEX,
							pinhole_rad=self.settings.PINHOLE_RAD,
							pinhole_shape=self.settings.PINHOLE_SHAPE,
							lat_dim=self.settings.LAT_DIM)

		return frame

	def simulate(self, pltshow=False):

		"""Realizes the trajectories in self.traj_df, painting
		   PSFs on frames

		Parameters
		----------
		"""

		nframes = self.settings.NFRAMES
		for i in range(nframes):
			this_traj_df = self.traj_df.loc[self.traj_df['frame'] == i]
			self.frames[i] = self.populate_frame(self.frames[i],
												 this_traj_df[['particle',\
												 'x','y','z','photons']])

	def add_noise(self):

		self.frames = add_noise_batch(self.frames,
									  self.settings.QUANT_EFF,
									  self.settings.SIGMA_DARK,
									  self.settings.BIT_DEPTH,
									  self.settings.SENSITIVITY,
									  self.settings.BASELINE,
									  self.settings.DYN_RANGE)



	def save(self, filename='hyperstack', corr_mat_3d=None):

		"""Save the movie to self.settings.OUTPUT_DIR

		Parameters
		----------
		"""

		if corr_mat_3d is not None:
			corr_movie=[]
			for i in range(len(self.frames)):
				fig,ax=plt.subplots(1,2)
				ax[0].imshow(corr_mat_3d[i], cmap='coolwarm',vmin=.5,vmax=1)
				ax[1].imshow(self.frames[i,0,0,:,:],cmap='coolwarm')
				corr_movie.append(plt2array(fig))
			corr_movie=np.array(corr_movie)
			imsave(self.settings.OUTPUT_DIR + 'corr_movie.tif', corr_movie)

		self.frames = self.frames/self.frames.max()
		self.frames = img_as_ubyte(self.frames)

		imsave(self.settings.OUTPUT_DIR + filename,
				self.frames,
				imagej=True,
				)
