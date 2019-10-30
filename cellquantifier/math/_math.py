import scipy.optimize as op
import numpy as np
import math

def fit_gaussian1d(x, y, amplitude):

	"""1D Gaussian fitting function

    Parameters
    ----------
    x: ndarray

    x0: float
        mean value of the gaussian

    amplitude: float
        initial guess for the amplitude of the gaussian

    Returns
    -------
    popt, pcov: ndarray
        optimal parameters and covariance matrix
    """

    def gaussian1d(x,x0,amplitude,sigma):

        y = amplitude*np.exp(-(x-x0)**2/(2*sigma**2))

        return y

	popt, pcov = curve_fit(lambda x, x0, sigma: gauss(x,amplitude,x0,sigma), x, y, p0 = [np.argmax(y), 25])

	return popt, pcov


def fit_poisson1d(x,y, scale):


	"""1D Scaled Poisson fitting function

    Parameters
    ----------
    x: 1d array

    y: 1d array
        raw data

    scale: float
        scaling factor for the poisson distribution

    Returns
    -------
    popt, pcov: ndarray
        optimal parameters and covariance matrix
    """

    def poisson(x, lamb, scale):

        return scale*(lamb**x/factorial(x))*np.exp(-lamb)

	popt, pcov = curve_fit(lambda x,lamb: poisson(x,lamb,scale), x,counts, p0=[np.argmax(counts)])

	return popt, pcov



def fit_msd(x,y, space='log'):

    """Mean Squared Dispacement fitting

    Parameters
    ----------
    t: 1d array

    y: 1d array
        raw data

    scale: float
        scaling factor for the poisson distribution

    Returns
    -------
    popt, pcov: ndarray
        optimal parameters and covariance matrix
    """

    def msd(t, D, alpha):

    	return 4*D*(t**alpha)

    def fit_msd_log(xdata, ydata):

        from scipy import stats

        xdata = [math.log(i) for i in xdata]
        ydata = [math.log(i) for i in ydata]

        slope, intercept, r, p, stderr = \
            stats.linregress(xdata,ydata)

        D = np.exp(intercept) / 4; alpha = slope
        popt = (D, alpha)
        return popt

    def fit_msd_linear(t, s):

    	popt, pcov = op.curve_fit(msd, t, s, bounds=(0, [np.inf, np.inf]))

    	return popt


def fit_spotcount(t, n, n0):

    """Spot count fitting function

    Parameters
    ----------
    t: 1d array

    n: 1d array
        number of spots

    n0: float
        intial number of spots

    Returns
    -------
    popt, pcov: ndarray
        optimal parameters and covariance matrix
    """

    def spot_count(t,n_ss,tau,n0):

    	return n_ss*(1-np.exp(-t/tau)) + n0

	popt, pcov = op.curve_fit(lambda t,n_ss,tau: spot_count(t,n_ss, tau, n0), t, n)

	return popt, popt


def fit_expdecay(t, s, b, t_c):

    """Exponential decay fitting function

    Parameters
    ----------
    t: 1d array

    n: 1d array
        number of spots

    n0: float
        intial number of spots

    Returns
    -------
    popt, pcov: ndarray
        optimal parameters and covariance matrix
    """


    def exp_decay(t,t_c,tau, b):

    	return b*(np.exp(-(t-t_c)/tau))

	popt, pcov = op.curve_fit(lambda t,tau: g(t,t_c,tau, b), t, s, p0=7)

	return popt, pcov

"""Sigmoid function definition and its corresponding caller"""

def sigmoid(t, c1, c2, n0, a):

	return a/(1+np.exp(-c1*(t-c2))) + n0

def sigmoid_fit(t, s, c2, n0):

	param_bounds=([0,1],[np.inf,1.5])

	popt, pcov = op.curve_fit(lambda t,a,c1: sig(t,c1,c2,a,n0), t, s, bounds=param_bounds)

	return popt, pcov
