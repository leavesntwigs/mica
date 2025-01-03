import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd

is_cfradial = True
is_mdv = False


# TODO: make a separate module/package for this ...
import matplotlib.colors as colors
from matplotlib.colors import to_rgb

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

color_scale_base = "/Users/brenda/git/lrose-displays/color_scales"
file_name = "zdr_color"
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
print(color_names)
print(edges)
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
    gg = 3
    norm = colors.BoundaryNorm(boundaries=edges, ncolors=len(color_names))
    norm.autoscale(edges)
# TODO: the edges are NOT uniform the everything steps by 1 except the last goes 12 to 20
    (zcmap, znorm) = colors.from_levels_and_colors(edges, color_scale_hex, extend='neither')
except ValueError as err:
    print("something went wrong first: ", err)

# A list of 2-element lists where the first element is the normalized color level value 
#  (starting at 0 and ending at 1), and the second item is a valid color string. 
#  (e.g. [[0, ‘green’], [0.5, ‘red’], [1.0, ‘rgb(0, 0, 255)’]])

extent = edges[-1] - edges[0]
normalized_edges = [ (i-edges[0])/extent for i in edges]
#colorscale_stacked = np.column_stack((normalized_edges, color_names))
#print(colorscale_stacked)


# use for cartesian data
def right_dim(var):
    if len(var.dims) > 2:
        return True
    return False


# NEXRAD
# TODO: make a default datatree structure
if is_cfradial:
    dirname = "/Users/brenda/data/for_mica/nexrad/output/20240510"
    filename = "cfrad.20240510_010615.273_to_20240510_011309.471_KBBX_SUR.nc"
    localfilename = dirname + "/" + filename
    datatree = xd.io.open_cfradial1_datatree(localfilename)
elif is_mdv:
    # cartesian data set
    ds_cart = xr.open_dataset("/Users/brenda/data/for_mica/ncf_20161006_191339.nc")
else:
    path = "/Users/brenda/data/for_mica/nexrad/"
    filename = "KBBX20240510_010615_V06"
    datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(path+filename)


# TODO: set default datatree and default filename, so that widgets display null
# at initial start up or on error?
# When file is chosen/changed, the other widgets update? because 
# all datatree arguments are replaced with get_datatree(filename)
# and filename is replaced with a function get_filename, which is bound to 
# select file_selector_widget? is that how it all flows through the pipes?

def get_datatree(file_full_path):
    new_datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(file_full_path)
    return new_datatree
    

def get_field_names(datatree):
# get field variables from a datatree ...
    sweep0 = datatree["/sweep_0"] # TODO make this more general and robust
    #print("dims: ", sweep0.dims)
    #print("data_vars: ", sweep0.data_vars)
    sweep0_data_vars = sweep0.data_vars
    fieldnames = []
    for k, v in sweep0_data_vars.items():
        #print(k, len(v.dims))
        if len(v.dims) >= 2:
            fieldnames.append(k)
    return fieldnames
#    return ['A', 'B', 'C']

def get_field_names(dataset):
# get field variables from a data set  ... 
    moments_cart = list([k for (k, v) in dataset.data_vars.items() if right_dim(v)])
    return moments_cart


minimum_number_of_gates = 50
def get_ranges(datatree):
    sweep = datatree['/sweep_0'] # TODO: what if this sweep doesn't exist?
    start_range = sweep.ray_start_range.data[0]
    gate_spacing = sweep.ray_gate_spacing.data[0]
    ngates = sweep.ray_n_gates.data[0]
    return [start_range+gate_spacing*minimum_number_of_gates,
        start_range+gate_spacing*int(ngates/2), start_range+gate_spacing*ngates] 

def map_range_to_index(max_range, rvals, gate_spacing, start_range):
    index = int((max_range - start_range)/gate_spacing)
    if index < minimum_number_of_gates:
        index = minimum_number_of_gates
    if index > len(rvals):
        index = len(rvals) - 1
    return index

def get_sweeps(datatree):
# get the number of sweeps from a datatree ...
    return len(datatree.groups) - 1


def show_status_open_file(dummy=1):
    return f'reading ...'

def styles(background):
    return {'background-color': background, 'padding': '0 10px'}

def get_plot(field, datatree):
    sweep = datatree['/sweep_0']
    rvals = sweep.range
    azvals = sweep.azimuth
    return f'nothing' # hv.Image(sweep.ZDR)
#     xs, ys = np.meshgrid(rvals, azvals)
#     return hv.Image(sweep[field], xs, ys)
   

xvals = np.linspace(-4, 0, 100)
yvals = np.linspace(4, 0, 100)
xs, ys = np.meshgrid(xvals, yvals)
# 
#


# def map_z_to_colors(Z, edges, colors):
#    replace with binary tree search
#     for z in Z:
#         found = False
#         pivot = int(len(edges)/2)
#         lower = 0
#         upper = len(edges)-1
#         while !found:
 
def find_slot(z, edges):

            if z < edges[pivot]:
                if upper == pivot:
                    return pivot
                upper = pivot
                pivot = int((pivot - lower)/2)
            elif z > edges[pivot]:
                if lower == pivot:
                    return pivet
                lower = pivot
                pivot = int((upper - pivot)/2)
            else:
                return pivot

# -------------- plotly ---------

import plotly.graph_objects as go
# import numpy as np

sweep = datatree['/sweep_0']

# Generate meshgrid data
max_range_index = 100
every_other_r = 10
every_other_az = 100
rvals = sweep.range[::every_other_r]  # [:max_range_index]
azvals = sweep.azimuth[0] # [::every_other_az]
field = 'ZDR'
fieldvar = sweep[field]
#z = fieldvar.data[::every_other_az,::every_other_r]  # :max_range_index]
z = fieldvar.data[0,::every_other_r]  # :max_range_index]
# z = fieldvar.data
R, Theta = np.meshgrid(rvals, azvals)

print('type(rvals): ', type(rvals))
print('shape z: ', fieldvar.data.shape)
print('some z values: ', z[::100])

# r = np.linspace(0, 1, 10)
# theta = np.linspace(0, 2*np.pi, 20)
# R, Theta = np.meshgrid(r, theta)
# Z = np.sin(R) * np.cos(Theta)
# Z = [3,5,7,9]
 
# map Z to colors
#Z_colors = map_z_to_colors(Z, colorscale)

# Convert to Cartesian coordinates
#X = R * np.cos(Theta)
#Y = R * np.sin(Theta)

# Create the figure
fig = go.Figure(data=[
    go.Scatterpolar(
#        r=rvals,
#        theta=azvals,
        r=R.flatten(),
        theta=Theta.flatten(),
        #r = rvals,  # [50000, 100000, 150000, 275000],
        #theta = azvals, # [30, 60, 90, 130],
        mode='markers',
        marker=dict(
            size=5,
            # set color to a numerical array 
            # color='blue',
            #color=Z_colors,
            #color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
            #   'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
            # color=['red', 'green', 'blue', 'orange'],
            color=z,  # Z,
            colorscale= [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']],
            cmin = -4, cmax = 20, # this OR cauto; still produces gradient colorscale
            # cauto = True,           # this OR cmin & cmax; still produces gradient colorscale
            # colorscale='viridis',
            # autocolorscale
            # cmax, cmid, cmin
            symbol='square-x',
            showscale=True,
            ## color should be based on a colorscale
        )
    )
])

#-------

# Generate sample data
#x = np.arange(-5, 5, 0.1)
#y = np.arange(-5, 5, 0.1)
#X, Y = np.meshgrid(x, y)
#Z = np.sin(X) * np.cos(Y)

# Create the heatmap
#fig = go.Figure(data=go.Heatmap(
#    z=Z,
#    x=x,
#    y=y,
#    colorscale='Viridis'
#))

#----

#x = np.random.randn(500)
#y = np.random.randn(500)+1

#xs, ys = np.meshgrid(rvals, azvals)
#field='ZDR'
#fig = go.Figure(data=go.Heatmap(x=R, y=Theta, z=Z,  # z=sweep[field],
#    colorscale='Viridis'
#    ))
#
#-------


# Customize the layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 3e5]),
        angularaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, 360, 45), 
            ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
        )
    )
)

fig.show()
