import pandas as pd

def sort_phys(df, sorters=None):
    """
	Wrapper for trackpy library functions (assign detection instances to particle trajectories)

	Parameters
	----------
	df : DataFrame
		DataFrame with column 'particle', 'dist_to_boundary', 'dist_to_53bp1'

	sorters: dict
		dictionary of filters

	Returns
	-------
	sorted_df : DataFrame
		DataFrame after sorting.

	Examples
	--------
    import pandas as pd
    from cellquantifier.phys import sort_phys

    sorters = {
        'DIST_TO_BOUNDARY': [-100, 0],
        'DIST_TO_53BP1' : [-5, 0],
    }
    df = pd.read_csv('cellquantifier/data/simulated_cell-physData.csv')
    sorted_df = sort_phys(df, sorters=sorters)
    print(df['particle'].unique())
    print(sorted_df['particle'].unique())
	"""

    sorted_df = df.copy()

    # """
	# ~~~~~~~~~~~~~~~~~Sort dist_to_boundary~~~~~~~~~~~~~~~~~
	# """

    if sorters['DIST_TO_BOUNDARY'] != None:
        df = df[(df['dist_to_boundary'] < min(sorters['DIST_TO_BOUNDARY'])) ^
                (df['dist_to_boundary'] > max(sorters['DIST_TO_BOUNDARY'])) ]
        bad_particles = df['particle']

    # """
	# ~~~~~~~~~~~~~~~~~Sort dist_to_53bp1~~~~~~~~~~~~~~~~~
	# """

    if sorters['DIST_TO_53BP1'] != None:
        df = df[(df['dist_to_53bp1'] < min(sorters['DIST_TO_53BP1'])) ^
                (df['dist_to_53bp1'] > max(sorters['DIST_TO_53BP1'])) ]
        bad_particles = pd.concat( [ bad_particles, df['particle'] ] )

    # """
	# ~~~~~~~~~~~~~~~~~Remove bad_particles from df~~~~~~~~~~~~~~~~~
	# """

    bad_particles = bad_particles.unique()
    for particle in bad_particles:
        sorted_df = sorted_df[ sorted_df['particle'] != particle ]

    return sorted_df
