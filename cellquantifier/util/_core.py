
def flatten(arr):

	"""

	Searches recursively for objects in an arbitarily nested array and
	returns the flattened array

	Pseudo code
	----------

	Parameters
	----------
	arr : nested list
		the nested list to recursively search for non-list objects

	Example
	----------
	"""

	def recurse(obj, out):

		if not isinstance(obj, list):
			out.append(obj)
		else:
			for x in obj:
				recurse(x, out)

		return out

	try:
		arr = list(arr)
	except TypeError:
		print('The input object was not of type list')

	flattened_arr = recurse(arr, out=[])

	return flattened_arr

def check_elem_type(iter, obj):
    """
    check if the type of all the elements in the iter is: obj.

    Pseudo code
    ----------
    1. If iter is not empty, and all elements are obj, return True.
    2. Otherwise, return False.

    Parameters
    ----------
    iter : iterable
        Iterable to be checked.

    obj : type
		The target object type.

    Returns
    -------
    Bool

    Examples
	--------
    from cellquantifier.io import *
    a = ['s', 'b', 'c']
    print(check_elem_type(a, str))
    """

    if iter:
        return all(isinstance(elem, obj) for elem in iter)
    else:
        return False

def check_elem_length(iter, len_num):
    """
    check if the length of all the elements in the iter equal to len_num.

    Pseudo code
    ----------
    1. If iter is not empty, and all elements' length equal to len_num, True.
    2. Otherwise, return False.

    Parameters
    ----------
    iter : iterable
        Iterable to be checked.

    len_num : int
		The target length.

    Returns
    -------
    Bool

    Examples
	--------
    from cellquantifier.io import *
    a = ['s', 'b', 'c']
    print(check_elem_length(a, 1))
    """

    if iter:
        return all(len(elem)==len_num for elem in iter)
    else:
        return False
