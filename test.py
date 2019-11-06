from cellquantifier.plot import plot_hist, scatter_bivariate, plot_cc_hist, plot_pie

import pandas as pd

# from cellquantifier.plot import plot_hist
# path1 = 'cellquantifier/data/test_Dalpha.csv'
# path2 = 'cellquantifier/data/test_Dalpha2.csv'
# df1 = pd.read_csv(path1, index_col=None, header=0)
# df2 = pd.read_csv(path2, index_col=None, header=0)
# labels = [r'D (nm$^2$/s)','Weight (a.u)']
# plot_hist(labels, damaged=df1['D'], control=df2['D'])

# from cellquantifier.plot import scatter_bivariate
# path = '/home/cwseitz/Desktop/190905_CtrBLM_Ctr9-dutp-fittDataFiltered.csv'
# df = pd.read_csv(path, index_col=None, header=0)
# df = df.groupby(['particle']).mean()
# y,x = df['D'], df['dist_to_com']
# scatter_bivariate(x,y, labels=['Distance to COM','D'], fit=True)

# from cellquantifier.plot import plot_hist
# path = '/home/cwseitz/Desktop/1162019_SpatialComparison/190905_CtrBLM_BLM8-dutp-fittDataFiltered.csv'
# labels = ['Distance to COM', 'Weight (a.u.)']
# df = pd.read_csv(path, index_col=None, header=0)
# plot_cc_hist(df[['dist_to_com', 'D']], nbins=3, labels=labels)

# from cellquantifier.plot import plot_hist
# path = '/home/cwseitz/Desktop/1162019_SpatialComparison/190905_CtrBLM_Ctr9-dutp-fittDataFiltered.csv'
# labels = [r'D (nm$^2$/s)', 'Weight (a.u.)']
# df = pd.read_csv(path, index_col=None, header=0)
# plot_hist(labels=labels, nbins=3, D=df['D'])

# from cellquantifier.plot import plot_hist
# path1 = '/home/cwseitz/Desktop/1162019_SpatialComparison/190905_CtrBLM_Ctr9-dutp-fittDataFiltered.csv'
# path2 = '/home/cwseitz/Desktop/1162019_SpatialComparison/190905_CtrBLM_BLM8-dutp-fittDataFiltered.csv'
# df1 = pd.read_csv(path1, index_col=None, header=0)
# df2 = pd.read_csv(path2, index_col=None, header=0)
#
# labels = ['slow', 'medium', 'fast']
# plot_pie(labels, nbins=3, control=df1['D'], damaged=df2['D'])

from cellquantifier.plot import plot_pie
path1 = 'cellquantifier/data/test_Dalpha.csv'
path2 = 'cellquantifier/data/test_Dalpha2.csv'
df1 = pd.read_csv(path1, index_col=None, header=0)
df2 = pd.read_csv(path2, index_col=None, header=0)
labels = ['slow', 'medium', 'fast']
plot_pie(labels, damaged=df1['D'], control=df2['D'])
