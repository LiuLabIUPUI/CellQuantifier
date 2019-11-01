def ax_format_1(ax):
    bottom, top = ax[0].get_ylim()
    if top > bottom:
        ax[0].set_ylim(top, bottom)
    ax.set_axis_off()
