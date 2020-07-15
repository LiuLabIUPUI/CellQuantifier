import scipy.optimize as op
import numpy as np
import math

# """
# ~~~~~~~~~~~~~~General Functions~~~~~~~~~~~~~~~~~~~~~
# """

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

def fit_linear(x, y):

	"""Perform linear regression on bivariate data

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

	from scipy import stats

	slope, intercept, r, p, stderr = \
		stats.linregress(x,y)

	return slope, intercept, r, p

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

# """
# ~~~~~~~~~~~~~~MSD Functions~~~~~~~~~~~~~~~~~~~~~
# """

def msd1(x, D, alpha):
	return 4*D*(x**alpha)

def msd2(x, D, alpha, c):
	return 4*D*(x**alpha) + c

def fit_msd1(x, y):
	popt, pcov = opt.curve_fit(msd1, x, y,
				  bounds=(0, [np.inf, np.inf]))
	return popt

def fit_msd1_log(x, y):
	x = [math.log(i) for i in x]
	y = [math.log(i) for i in y]

	slope, intercept, r, p, stderr = \
		stats.linregress(x,y)

	D = np.exp(intercept) / 4; alpha = slope
	popt = (D, alpha)

	return popt

def fit_msd2(x, y):
	popt, pcov = opt.curve_fit(msd2, x, y,
				  bounds=(0, [np.inf, np.inf, np.inf]),
				  )
	return popt

# """
# ~~~~~~~~~~~~~~Misc Functions~~~~~~~~~~~~~~~~~~~~~
# """

def spot_count(x,a,tau,c):

	return a*(1-np.exp(-x/tau)) + c

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

	c = y[0]
	popt, pcov = op.curve_fit(lambda x,a,tau: spot_count(x,a,tau,c), x, y)
	popt = [*popt, c]

	return popt

def interpolate_lin(x_discrete, f_discrete, resolution=100, pad_size=0):

	"""Numpy wrapper for performing linear interpolation

	Parameters
	----------
	x_discrete: 1d ndarray
		discrete domain

	y: 1d ndarray
		discrete function of x_discrete

	Returns
	-------

	"""

	min_center, max_center = x_discrete[0], x_discrete[-1]
	x_cont = np.linspace(min_center, max_center, resolution)

	f_cont = np.interp(x_cont, x_discrete, f_discrete)
	if pad_size > 0:

		x_pad = np.linspace(min_center-pad_size, min_center, pad_size)
		f_cont_pad = np.full(pad_size, 0)
		x_cont = np.concatenate((x_pad, x_cont), axis=0)
		f_cont = np.concatenate((f_cont_pad, f_cont), axis=0)

	return x_cont, f_cont

def ransac_polyfit(x_array_1d, y_array_1d, poly_deg,
                  min_sample_num,
                  residual_thres,
                  max_trials,
                  stop_sample_num=np.inf,
                  random_seed=None):
    """
    Run the RANSAC polynomial fitting.

    Parameters
    ----------
    x_array_1d : ndarray
        xdata.
    x_array_1d : ndarray
        ydata.
    poly_deg : int
        Polynomial degree.
    min_sample_num : int
        Minimum samples numbers needed for RANSAC fitting.
    residual_thres : float
        Residual threshold for RANSAC fitting.
    max_trials : int
        Maximum trials number for RANSAC fitting.
    stop_sample_num : int, optional
        Stop sample number for RANSAC fitting.
    random_seed : float, optional
        Random seed for RANSAC fitting.

    Returns
    -------
    params_tuple_1d: tuple
        1d tuple of the output parameters.

    Examples
    --------
    import numpy as np
    from cellquantifier.qmath.ransac import ransac_polyfit
    a = np.array(range(10))
    b = 1 + 2*a + 3*a ** 2
    params = ransac_polyfit(x_array_1d=a, y_array_1d=b, poly_deg=2,
                    min_sample_num=5, residual_thres=2, max_trials=100)
    print(params)
    """

    # """
    # ~~~~~~~Generate a list. xdata as the 1st col, ydata as the 2nd col~~~~~~~
    # """

    data = np.zeros((len(x_array_1d), 2))
    data[:,0] = x_array_1d
    data[:,1] = y_array_1d
    datalist = list(data)

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Do the ransac fitting~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    best_inlier_count = 0
    best_model = None
    best_err = None
    random.seed(random_seed)
    for i in range(max_trials):
        sample = random.sample(datalist, int(min_sample_num))
        sample_ndarray = np.array(sample)
        xdata_s = sample_ndarray[:,0]
        ydata_s = sample_ndarray[:,1]
        poly_params = np.polyfit(xdata_s, ydata_s, poly_deg)
        p = np.poly1d(poly_params)

        inlier_count = 0
        for j in range(len(data)):
            curr_x = data[j,0]
            rms = np.abs(p(curr_x) - data[j, 1])
            if rms < residual_thres:
                inlier_count = inlier_count + 1

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = poly_params
            if inlier_count > stop_sample_num:
                break

    # """
    # ~~~~~~~~~~~~~~~~~~~~~Print the ransac fitting summary~~~~~~~~~~~~~~~~~~~~~
    # """

    print("#" * 30)
    print("Iteration_num: ", i+1)
    print("Best_inlier_count: ", best_inlier_count)
    print("Best_model: ", best_model)
    print("#" * 30)

    params_tuple_1d = best_model

    return params_tuple_1d
