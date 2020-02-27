def check_list_type(list, obj):
    """
    check if the type of all the elements in the list is: obj.

    Pseudo code
    ----------
    1. If list is not empty, and all elements are obj, return True.
    2. Otherwise, return False.

    Parameters
    ----------
    list : list
        List to be checked.

    obj : type
		The target object type.

    Returns
    -------
    Bool

    Examples
	--------
    from cellquantifier.io import *
    a = ['s', 'b', 'c']
    print(check_list_type(a, str))
    """

    if list:
        return all(isinstance(elem, obj) for elem in list)
    else:
        return False
