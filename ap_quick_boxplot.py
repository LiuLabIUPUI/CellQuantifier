import pandas as pd
from cellquantifier.publish._fig_quick_boxplot import *

df = pd.read_csv('/home/linhua/Desktop/xx/210301-physDataMerged.csv')
fig_quick_boxplot(df)
