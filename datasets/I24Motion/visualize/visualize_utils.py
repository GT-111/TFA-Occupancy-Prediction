from matplotlib.colors import LinearSegmentedColormap
def get_cmap(colors):
    
    return LinearSegmentedColormap.from_list('while-colors', colors, N=256)