import scipy.optimize as op
import numpy as np
import math

def fit_gaussian1d(x, y):

	"""1D Gaussian fitting function

	Parameters
	----------
	x: 1d ndarray
		raw x data

	y: 1d ndarray
		raw y data

	Returns
	-------
	popt, pcov: 1d ndarray
		optimal parameters and covariance matrix
	"""

	popt, pcov = curve_fit(gaussian1d, x, y)

	return popt, pcov

def gaussian1d(x,x0,amp,sigma):

	y = amp*np.exp(-(x-x0)**2/(2*sigma**2))

	return y


def fit_poisson1d(x,y):


	"""1D Scaled Poisson fitting function

	Parameters
	----------
	x: 1d ndarray
		raw x data

	y: 1d ndarray
		raw y data

	scale: float
		scaling factor for the poisson distribution

	Returns
	-------
	popt, pcov: ndarray
		optimal parameters and covariance matrix
	"""

	popt, pcov = curve_fit(poisson1d,x,y)

	return popt, pcov

def poisson1d(x, lambd, scale):

	return scale*(lambd**x/factorial(x))*np.exp(-lambd)


def fit_msd(x,y, space='log'):

	"""Mean Squared Dispacement fitting

	Parameters
	----------
	x: 1d array
		raw x data

	y: 1d array
		raw y data

	space: string
		'log' for fitting in log space (default)
		'linear' for sitting in linear space

	Returns
	-------
	popt, pcov: 1d ndarray
		optimal parameters and covariance matrix
	"""

	def fit_msd_log(x, y):

		from scipy import stats

		x = [math.log(i) for i in x]
		y = [math.log(i) for i in y]

		slope, intercept, r, p, stderr = \
			stats.linregress(x,y)

		D = np.exp(intercept) / 4; alpha = slope
		popt = (D, alpha)
		return popt

	def fit_msd_linear(x, y):

		popt, pcov = op.curve_fit(msd, x, y, bounds=(0, [np.inf, np.inf]))

		return popt

	if space == 'log':
		popt = fit_msd_log(x,y)
	elif space == 'linear':
		popt = fit_msd_linear(x,y)

	return popt

def msd(x, D, alpha):

	return 4*D*(x**alpha)


def fit_spotcount(x, y):

	"""Spot count fitting function

	Parameters
	----------
	x: 1d ndarray
		raw x data

	y: 1d ndarray
		raw y data

	Returns
	-------
	popt, pcov: 1d ndarray
		optimal parameters and covariance matrix
	"""

	def spot_count(x,a,b,c):

		return a*(1-np.exp(-x/b)) + c

	popt, pcov = op.curve_fit(spot_count,x,y)

	return popt, popt


def fit_expdecay(x,y):

	"""Exponential decay fitting function

	Parameters
	----------
	x: 1d ndarray
		raw x data

	y: 1d ndarray
		raw y data

	Returns
	-------
	popt, pcov: 1d ndarray
		optimal parameters and covariance matrix
	"""

	def exp_decay(x,a,b,c):

		return a*(np.exp(-(x-b)/c))

	popt, pcov = op.curve_fit(exp_decay, x, y)

	return popt, pcov

def fit_sigmoid(x,y):

	"""Sigmoid fitting function

	Parameters
	----------
	x: 1d ndarray
		raw x data

	y: 1d ndarray
		raw y data

	Returns
	-------
	popt, pcov: 1d ndarray
		optimal parameters and covariance matrix
	"""

	def sigmoid(x, a, b, c, d):

		return a/(1+np.exp(-b*(x-c))) + d

	param_bounds=([0,1],[np.inf,1.5])

	popt, pcov = op.curve_fit(sigmoid, x, y)

	return popt, pcov
