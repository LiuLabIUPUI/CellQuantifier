import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from ..plot.plotutil import *

def anim_blob(df, tif):
    anim_tif = []

    for i in range(len(tif)):
        print("Animate frame %d" % i)
        curr_df = df[ df['frame']==i ]

        fig, ax = plt.subplots(figsize=(9,9))

        ax.imshow(tif[i], cmap='gray', aspect='equal')
        anno_blob(ax, curr_df, marker='^', markersize=10, plot_r=False,
                    color=(0,0,1))

        # """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # """
        font = {'family': 'arial', 'weight': 'bold','size': 16}
        scalebar = ScaleBar(0.163, 'um', location = 'upper right',
        	font_properties=font, box_color = 'black', color='white')
        scalebar.length_fraction = .3
        scalebar.height_fraction = .025
        ax.add_artist(scalebar)

        curr_plt_array = plot_end(fig, pltshow=False)
        anim_tif.append(curr_plt_array)

    anim_tif = np.array(anim_tif)

    return anim_tif
