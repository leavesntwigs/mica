# include in simple_plot_app.py or use in Jupyter notebook

# from dash import Dash, dash_table, html, dcc, Input, Output, ctx, State, ALL, MATCH, Patch,  callback, no_update
# import dash_ag_grid as dag
# import plotly.express as px
# import plotly.graph_objects as go

import matplotlib                                
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import xarray as xr
import xradar as xd
#import cmweather
import numpy as np

# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")


# TODO: make a separate module/package for this ...
# then, make it available like this ...
# from colormapf import color_conversion_base, etc.
# 
import matplotlib.colors as colors
from matplotlib.colors import to_rgb

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
            # print(x)
            if len(x) == 3:
                color_names.append(x[2].lower())
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
    import matplotlib.cm as cm
    
    colormap_function = cm.get_cmap('viridis')
    edges =  np.linspace(0, 1, 256) # how to set the number of colors (256)?
    colors_rgb = colormap_function(edges) # Get 256 colors from the colormap
    print(colors_rgb.shape) # Output: (256, 4) - 256 colors with RGBA values
    return convert_to_go_colorscale(edges, colors_hex)


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
        return colorscale_for_go

    # otherwise, return error


## how to use in a polar plot
#def plot_data_scatter(selected_field, 
#    datatree_sweep, 
#    max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
#    print("inside plot_data_scatter")
#    if datatree_sweep == None:
#        print("datatree_sweep is None") 
#    sweep = datatree_sweep
#    rvals = sweep.range
#    azvals = sweep.azimuth
#    # Generate data in polar coordinates
#    test_data = False # True
#    if test_data:
#        theta = np.linspace(0, 2 * np.pi, 360)
#        r = np.linspace(0, 1, 100)
#        R, Theta = np.meshgrid(r, theta)
#        Z = np.sin(R) * np.cos(Theta)
#    else:
#        print("all good")
#        max_range_index = 100  
#        # the azimuth need to be sorted into ascending order
#        theta = np.linspace(0, 360, 360) # azvals
#        # r = np.linspace(0,1, max_range_index)
#        r = rvals[:max_range_index]
#        R, Theta = np.meshgrid(r, theta)
#        fieldvar = sweep[selected_field] # sweep[field]
#        #                              (nrows, ncolumns)
#        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
#        z = fieldvar.data[:,:max_range_index]
#        print("r: ", r.data)
#        print("azvals: ", azvals.data)
#        print("z: ", z)
#        print("fieldvar.data dims: ", fieldvar)
#        print("Theta flattened: ", Theta.flatten())
#        scale_factor = 360/len(azvals) # or min_distance_between_rays * 360???
#        z_bin_sort = np.full((360,max_range_index), fill_value=-3200)
#        for i in range(0,360):
#            raw_az = azvals[i]
#            new_z_index = int(raw_az/scale_factor)
#            if (new_z_index >= 360):
#                new_z_index -= 360
#            z_bin_sort[new_z_index] = z[i]
#        Z = np.nan_to_num(z_bin_sort, nan=-32)
#
#*****
#    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
#    # use matplotlib function to creatte a colormp
#    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
#    print("color_scale_hex: ", color_scale_hex)
#    print("edges: ", edges)
#    print("colors_norm: ", colors_norm)
#    print("edges_norm: ", edges_norm)
#    colorscale_for_go = convert_to_go_colorscale(edges, color_scale_hex)
#    print("colorscale_for_go: ", colorscale_for_go)
#
#*****    
#    #  Create a polar plot
#
#    fig = go.Figure(data=
#        go.Scatterpolar(
#            r = R.flatten(), 
#            theta = Theta.flatten(),
#            mode = 'markers',
#            marker=dict(
#                size=5,
#                # set color to a numerical array 
#                #color=Z_colors,
#                #color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
#                #   'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
#                # color=['red', 'green', 'blue', 'orange'],
#                color=Z.flatten(),
#                # range_color=[-4, 20],
#***                colorscale=colorscale_for_go,
##                   [(0.00, "red"),   (0.33, "red"),
##                    (0.33, "green"), (0.66, "green"),
##                    (0.66, "blue"),  (1.00, "blue")],
#                # colorscale= [[0, 'blue'], [0.5, 'green'], [1.0, 'red']],
#***                cmin = -4, cmax = 20, # this OR cauto; still produces gradient colorscale
#                # cauto = True,           # this OR cmin & cmax; still produces gradient colorscale
#                # cmax, cmid, cmin
#                symbol='square-x',
#                showscale=True,
#                ## color should be based on a colorscale
#            )
#        ))
#
#    fig.update_layout(
#       polar = dict(
#          angularaxis = dict(
#             rotation=90,
#             direction="clockwise"
#          )
#       )
#    )    
#    fig.update_layout(clickmode='event+select')
#
#    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
#    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
##    psm = ax.pcolormesh(Theta, R, Z,
##        cmap=cmap,
##        vmin=edges_norm[0], vmax=edges_norm[-1],
##        # norm=norm,
##        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
##        rasterized=True, shading='nearest')
##    # make the top 0 degrees and the angles go clockwise
##    ax.set_theta_direction(-1)
##    ax.set_theta_offset(np.pi / 2.0)
##    # try to set the radial lines
##    rtick_locs = np.arange(0,25000,5000)
##    rtick_labels = ['%.1f'%r for r in rtick_locs]
##    ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
##    #
##    fig.colorbar(psm, ax=ax)
##    # Add a title
##    plt.title('Quadmesh on Polar Coordinates (matplotlib): ' + selected_field)
#
#    print('**** done with plot_data_scatter  ***')
#
#    return fig
#
