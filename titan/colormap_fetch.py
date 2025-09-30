# include in simple_plot_app.py or use in Jupyter notebook

import matplotlib                                
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import numpy as np



# TODO: make a separate module/package for this ...
# then, make it available like this ...
# from colormapf import color_conversion_base, etc.
# 
import matplotlib.colors as colors
from matplotlib.colors import to_rgb

def to_rgb(v):
    return int(v*255)

#
# NOTE:  depends on color map file to convert X11 color names to hex
#
color_conversion_base = "/Users/brenda/git/mica"
file_name = "x11_colors_map.txt"
conversion_file = color_conversion_base + "/" + file_name
x11_color_name_map = {}  # it is a dictionary
# read the color name to hex conversion file
f = open(conversion_file, "r")
for line in f.readlines():
    x = line.split()
    x11_color_name_map[x[0]]=x[1]

def normalize_colormap(edges, colors):
    nsteps = int(edges[-1] - edges[0] + 1)
    new_edges = np.linspace(edges[0], edges[-1], nsteps)
    new_colors = []
    for i in range(0, len(edges)-1):
        for ii in range(int(edges[i]), int(edges[i+1])):
            new_colors.append(colors[i])
    return (new_edges, new_colors)

# example using lrose-displays ...
# lrose-displays file format:
# MIN  MAX     NAME (name is ascii text or hex #rrggbb )
# 0      5    yellow

# TODO: make color_scale_base_dir  an input setting
def fetch_lrose_displays_color_scale(color_scale_name, color_scale_base_dir=None):
    if color_scale_base_dir == None:
        color_scale_base = "/Users/brenda/git/lrose-displays/color_scales"
    else:
        color_scale_base = color_scale_base_dir
    file_name = color_scale_name    # "zdr_color"    
    color_scale_file = color_scale_base + "/" + file_name
    color_names = []
    edges = []
    # read the color map file
    f = open(color_scale_file, "r")
    for line in f.readlines():
        if line[0] != '#':
            # print(line)
            x = line.split()
            print(x)
            if len(x) >= 3:  # TODO sometimes the color name has multiple words!!! dark slate blue!!!
                color_names.append(" ".join(x[2:]).lower())
                if len(edges) == 0:
                    edges.append(float(x[0]))
                edges.append(float(x[1]))
    # add the ending edge
    # display(color_names)
    # display(edges)
    # convert the X11 color names to hex
    color_scale_hex = []
    for cname in color_names:
        if cname in x11_color_name_map:
            color_scale_hex.append(x11_color_name_map[cname]) 
        else:
            color_scale_hex.append(colors.to_hex(cname))

    # convert color names to rgb
    #rgb_color = to_rgb("dodgerblue")
    # define color map (matplotlib.colors.ListedColormap)
    try:
        # is this used???
        norm = colors.BoundaryNorm(boundaries=edges, ncolors=len(color_names))
        norm.autoscale(edges)
    # TODO: the edges are NOT uniform the everything steps by 1 except the last goes 12 to 20
        (zcmap, znorm) = colors.from_levels_and_colors(edges, color_scale_hex, extend='neither')
    except ValueError as err:
        print("something went wrong first: ", err)
    print("edges: ")
    print(edges)
    print("colors: ", color_scale_hex)
    return edges, color_scale_hex 

# function:  convert_to_go_colorscale takes edges and colors in hex
#               and converts this information to a colormap format 
#               recognized by Dash graph objects,
#               scatterpolar.
# 
#                colorscale=[(0.00, "red"),   (0.33, "red"),
#                    (0.33, "green"), (0.66, "green"),
#                    (0.66, "blue"),  (1.00, "blue")],
# color_scale_hex:  ['#483d8b', '#000080', '#0000ff', '#0000cd', '#87ceeb', '#006400', '#228b22', '#9acd32', '#bebebe', '#f5deb3', '#ffd700', '#ffff00', '#ff7f50', '#ffa500', '#c71585', '#ff4500', '#ff0000']
# edges:  [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 20.0]
# use for lrose-display and self-defined coloscales
def convert_to_go_colorscale(edges, colors_hex):
    colorscale=[]
    max = edges[-1]
    min = edges[0]
    edge_range = np.abs(max-min)
    print("max = ", max, " min = ", min, " edge_range = ", edge_range)
    for i in range(len(edges)-1):
       low = (edges[i]-min)/edge_range
       high = (edges[i+1]-min)/edge_range
       c = colors_hex[i]
       colorscale.append((low, c))
       colorscale.append((high, c))
    return colorscale 
    
# example using matplotlib color maps ...
def convert_to_go_colorscalei_matplotlib(color_scale_name):
# example with colorblind friendly maps (reference Py-ART)
    #plot_color_gradients(
        #"Colorblind Friendly",
        #["LangRainbow12", "HomeyerRainbow", "balance", "ChaseSpectral", "SpectralExtended"],
    #)
    import numpy as np
    #import matplotlib.cm as cm
    
    colormap_function = plt.get_cmap(color_scale_name)
    edges =  np.linspace(0, 1, 20) # how to set the number of colors (256)?
    colors_rgb = colormap_function(edges) # Get 256 colors from the colormap

    # need list of tuples [(edge, color), ...]
    clist = []
    i = 0
    for rgba in colors_rgb.tolist():
        rgb_tuple = tuple(map(to_rgb, rgba[0:3]))
        clist.append([edges[i], 'rgb'+str(rgb_tuple)])
        i += 1

    print(clist) # Output: (256, 4) - 256 colors with RGBA values
    # RGBA values are 
    # rgba(red, green, blue, alpha)
    # The alpha parameter is a number between 0.0 (fully transparent) and 1.0 (not transparent at all):
   
    return clist 
    # return convert_to_go_colorscale(edges, colors_hex)


# here ...
#>>> def to_rgb(v):
#...     return int(v*255)
#
#>>> for rgba in c_rgb.tolist():
#...         list(map(to_rgb, rgba[0:3]))

#    The 'colorscale' property is a colorscale and may be
#    specified as:
#      - A list of colors that will be spaced evenly to create the colorscale.
#        Many predefined colorscale lists are included in the sequential, diverging,
#        and cyclical modules in the plotly.colors package.
#      - A list of 2-element lists where the first element is the
#        normalized color level value (starting at 0 and ending at 1),
#        and the second item is a valid color string.
#        (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
#      - One of the following named colorscales:
#            ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
#             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
#             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
#             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
#             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
#             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
#             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
#             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
#             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
#             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
#             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
#             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
#             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
#             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
#             'ylorrd'].
#        Appending '_r' to a named colorscale reverses it.
#


# main entry point here ...
def fetch(color_map_name):
    # is it a matplotlib name?
    
    if color_map_name in plt.colormaps():
        print("matplotlib knows the color scale")
        return convert_to_go_colorscalei_matplotlib(color_map_name)
    # is it an lrose-display name?
    # if ???
    else:
        print("must be an lrose-display color scale")
        #fetch_lrose_displays_color_scale(color_map_name)
        edges, color_scale_hex = fetch_lrose_displays_color_scale(color_map_name)
    # (edges_norm, colors_norm) = colormap_fetch.normalize_colormap(edges, color_scale_hex)
    # cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
        # print("color_scale_hex: ", color_scale_hex)
        # print("edges: ", edges)
        # print("colors_norm: ", colors_norm)
        # print("edges_norm: ", edges_norm)
        colorscale_for_go = convert_to_go_colorscale(edges, color_scale_hex)
        print("color scale: ")
        print(colorscale_for_go)
        return colorscale_for_go

    # otherwise, return error


