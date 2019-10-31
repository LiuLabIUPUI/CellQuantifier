def ax_format_1(ax, image):
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    ax.set_axis_off()
