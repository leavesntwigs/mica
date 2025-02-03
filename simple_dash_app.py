from dash import Dash, dash_table, html, dcc, Input, Output, State, callback
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
# import dash_ag_grid as dag                       
# import dash_bootstrap_components as dbc          
import pandas as pd                              

import matplotlib                                
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import xarray as xr
import xradar as xd
#import cmweather
import numpy as np

import subprocess
import os

# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['cBLwgT.css']

#app = Dash(__name__, external_stylesheets=external_stylesheets)
app = Dash(__name__, )

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')


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
    gg = 3
    norm = colors.BoundaryNorm(boundaries=edges, ncolors=len(color_names))
    norm.autoscale(edges)
# TODO: the edges are NOT uniform the everything steps by 1 except the last goes 12 to 20
    (zcmap, znorm) = colors.from_levels_and_colors(edges, color_scale_hex, extend='neither')
except ValueError as err:
    print("something went wrong first: ", err)


#                colorscale=[(0.00, "red"),   (0.33, "red"),
#                    (0.33, "green"), (0.66, "green"),
#                    (0.66, "blue"),  (1.00, "blue")],
# color_scale_hex:  ['#483d8b', '#000080', '#0000ff', '#0000cd', '#87ceeb', '#006400', '#228b22', '#9acd32', '#bebebe', '#f5deb3', '#ffd700', '#ffff00', '#ff7f50', '#ffa500', '#c71585', '#ff4500', '#ff0000']
# edges:  [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 20.0]
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
    

# use for cartesian data
def right_dim(var):
    if len(var.dims) > 2:
        return True
    return False

def get_field_names(datatree):
    print('inside get_field_names')
# get field variables from a datatree ...
    sweep0 = datatree["/sweep_0"] # TODO make this more general and robust
    #print("dims: ", sweep0.dims)
    print("data_vars: ", sweep0.data_vars)
    sweep0_data_vars = sweep0.data_vars
    fieldnames = []
    for k, v in sweep0_data_vars.items():
        #print(k, len(v.dims))
        if len(v.dims) >= 2:
            fieldnames.append(k)
    return fieldnames
#     return ['A', 'B', 'C']


# NEXRAD
# TODO: make a default datatree structure
if is_cfradial:
    dirname = "/Users/brenda/data/for_mica/nexrad/output/20240510"
    filename = "cfrad.20240510_010615.273_to_20240510_011309.471_KBBX_SUR.nc"
    localfilename = dirname + "/" + filename
    datatree = xd.io.open_cfradial1_datatree(localfilename)
    print('before blue')
    field_names_8 = get_field_names(datatree)
    print('field_names_8 = ', field_names_8)
    print('after blue')
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
    

#def get_field_names(dataset):
## get field variables from a data set  ... 
#    moments_cart = list([k for (k, v) in dataset.data_vars.items() if right_dim(v)])
#    return moments_cart


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

# '/sweep_8'
sweep_names = ['1','2','3']

#file_selector_widget = pn.widgets.FileSelector('~/data')

#card = pn.Card(file_selector_widget, title='Step 1. Choose Data File', styles={'background': 'WhiteSmoke'})

# x = pn.widgets.IntSlider(name='sweep', start=0, end=get_sweeps(datatree)-1)
#background = pn.widgets.ColorPicker(name='Background', value='lightgray')
# field_names_widget = pn.widgets.Select(name="field", options=get_field_names(datatree))
# open_file_widget = pn.widgets.Button(name="open/read file?", button_type='primary')


def show_selected_field(field, x):
    return f'selected field is {field} sweep is {x}'

# use with pn.pane.Markdown
#def show_selected_file(file_name):
#    if len(file_name) <= 0:
#        return f'Select a file to open'
#    else:
#        return f'You selected {file_name}'
    

# use with pn.pane.HoloViews
def show_selected_file(file_name):
    if len(file_name) <= 0:
        if is_mdv:
           height_initial_value = [0]
        else:
           height_initial_value = ['/sweep_8']
        return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(max_range=[25000,30000], beta=height_initial_value, field=['ZDR', 'DBZH', 'RHOHV'])
    else:
        if is_mdv:
            # cartesian data set; use dataset structure
            ds_cart = xr.open_dataset(file_name[0])
            fields = get_field_names(ds_cart)
            heights = np.arange(0,len(ds_cart.z0))  # [0,1,2,3] # ds_cart.z0.data # get_sweeps(ds_cart)
            return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(
                max_range=[100,200,300,400, 500, 600, 700],
                beta=heights,  
                field=fields)
        else:
            # polar data; use datatree structure
            if is_cfradial:
                datatree = xd.io.open_cfradial1_datatree(file_name[0])
            else:
                datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(file_name[0])
            fields = get_field_names(datatree)
            sweeps = get_sweeps(datatree)
            print(datatree.groups)
            sweep_names = [name for name in datatree.groups if 'sweep' in name] 
            return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(
                max_range=get_ranges(datatree), # [100,200,300,400, 500, 600, 700], 
                beta=sweep_names,  # datatree.groups,
                field=fields) # , dtree=datatree)
            # return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(max_range=[1,2,3], beta=[0.1, 1.0, 2.5], field=fields) # , dtree=datatree)


def styles(background):
    return {'background-color': background, 'padding': '0 10px'}


print('result of get_field_names: ', get_field_names(datatree)) 

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

# cid = fig.canvas.mpl_connect('button_press_event', onclick)

def find_nearest_index(array, value):
    # float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_field_data_blah(field_name='ZDR', theta=50.0):
    print("first line of get_field_data")
    float(theta)
    sweep = datatree['/sweep_8']
    print("field_name: ", field_name, theta)
    fieldvar = sweep[field_name]
    azvals = sweep.azimuth
    az_index = find_nearest_index(azvals, theta)   
    z = fieldvar.data[az_index,:] # fieldvar.data[az_index,:]
    print("get_field_data: ", z[:5])
    return z

def get_ranges():
    sweep = datatree['/sweep_8']
    return sweep.range.data

def plot_data_polly(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    # uses global datatree ...
    sweep = datatree['/sweep_8']
    rvals = sweep.range
    azvals = sweep.azimuth
    # Generate data in polar coordinates
    test_data = False # True
    if test_data:
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 1, 100)
        R, Theta = np.meshgrid(r, theta)
        Z = np.sin(R) * np.cos(Theta)
    else:
        max_range_index = 100
# the azimuth need to be sorted into ascending order
        theta = np.linspace(0, 2 * np.pi, 360) # azvals 
        # r = np.linspace(0,1, max_range_index)
        r = rvals[:max_range_index]
        R, Theta = np.meshgrid(r, theta)
        fieldvar = sweep[selected_field] # sweep[field]
        #                              (nrows, ncolumns)
        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        z = fieldvar.data[:,:max_range_index]
        scale_factor = 360/len(azvals) # or min_distance_between_rays * 360???
        z_bin_sort = np.full((360,max_range_index), fill_value=-3200)
        for i in range(0,360):
            raw_az = azvals[i]
            new_z_index = int(raw_az/scale_factor)
            if (new_z_index >= 360):
                new_z_index -= 360
            z_bin_sort[new_z_index] = z[i]
        Z = np.nan_to_num(z_bin_sort, nan=-32)
        # Z = np.sin(R) * np.cos(Theta)
    # return  hv.QuadMesh((R, Theta, Z)).options(projection='polar', cmap='seismic',)
    #  Create a polar plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
    # Plot the quadmesh
    #            (X(column), Y(row), Z(row,column))
    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
    psm = ax.pcolormesh(Theta, R, Z,
        cmap=cmap,
        vmin=edges_norm[0], vmax=edges_norm[-1],
        # norm=norm,
        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
        rasterized=True, shading='nearest')
    # make the top 0 degrees and the angles go clockwise
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    # try to set the radial lines
    rtick_locs = np.arange(0,25000,5000)
    rtick_labels = ['%.1f'%r for r in rtick_locs]
    ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
    #
    fig.colorbar(psm, ax=ax)
    # Add a title
    plt.title('Quadmesh on Polar Coordinates (matplotlib): ' + selected_field)


    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib2 = f'data:image/png;base64,{fig_data}'

    print('**** done with polar 2 ***')

    return fig_bar_matplotlib2


def plot_data_scatter(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside plot_data_scatter")
    # uses global datatree ...
    sweep = datatree['/sweep_8']
    rvals = sweep.range
    azvals = sweep.azimuth
    # Generate data in polar coordinates
    test_data = False # True
    if test_data:
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 1, 100)
        R, Theta = np.meshgrid(r, theta)
        Z = np.sin(R) * np.cos(Theta)
    else:
        max_range_index = 100  
# the azimuth need to be sorted into ascending order
        theta = np.linspace(0, 360, 360) # azvals
        # r = np.linspace(0,1, max_range_index)
        r = rvals[:max_range_index]
        R, Theta = np.meshgrid(r, theta)
        fieldvar = sweep[selected_field] # sweep[field]
        #                              (nrows, ncolumns)
        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        z = fieldvar.data[:,:max_range_index]
        print("r: ", r.data)
        print("azvals: ", azvals.data)
        print("z: ", z)
        print("fieldvar.data dims: ", fieldvar)
        print("Theta flattened: ", Theta.flatten())
        scale_factor = 360/len(azvals) # or min_distance_between_rays * 360???
        z_bin_sort = np.full((360,max_range_index), fill_value=-3200)
        for i in range(0,360):
            raw_az = azvals[i]
            new_z_index = int(raw_az/scale_factor)
            if (new_z_index >= 360):
                new_z_index -= 360
            z_bin_sort[new_z_index] = z[i]
        Z = np.nan_to_num(z_bin_sort, nan=-32)

    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
    print("color_scale_hex: ", color_scale_hex)
    print("edges: ", edges)
    print("colors_norm: ", colors_norm)
    print("edges_norm: ", edges_norm)
    colorscale_for_go = convert_to_go_colorscale(edges, color_scale_hex)
    print("colorscale_for_go: ", colorscale_for_go)
    
    #  Create a polar plot

    fig = go.Figure(data=
        go.Scatterpolar(
            r = R.flatten(), # [0.5,1,2,2.5,3,4],
            theta = Theta.flatten(), # [35,70,120,155,205,240],
            # startangle=0,
            # direction="counterclockwise",
            mode = 'markers',
            marker=dict(
                size=5,
                # set color to a numerical array 
                # color='blue',
                #color=Z_colors,
                #color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
                #   'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
                # color=['red', 'green', 'blue', 'orange'],
                color=Z.flatten(),
                # range_color=[-4, 20],
                colorscale=colorscale_for_go,
#                   [(0.00, "red"),   (0.33, "red"),
#                    (0.33, "green"), (0.66, "green"),
#                    (0.66, "blue"),  (1.00, "blue")],
                # colorscale= [[0, 'blue'], [0.5, 'green'], [1.0, 'red']],
                cmin = -4, cmax = 20, # this OR cauto; still produces gradient colorscale
                # cauto = True,           # this OR cmin & cmax; still produces gradient colorscale
                # colorscale='viridis',
                # autocolorscale
                # cmax, cmid, cmin
                symbol='square-x',
                showscale=True,
                ## color should be based on a colorscale
            )
        ))

    fig.update_layout(
       polar = dict(
          angularaxis = dict(
             rotation=90,
             direction="clockwise"
          )
       )
    )    
    fig.update_layout(clickmode='event+select')

    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
#    psm = ax.pcolormesh(Theta, R, Z,
#        cmap=cmap,
#        vmin=edges_norm[0], vmax=edges_norm[-1],
#        # norm=norm,
#        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
#        rasterized=True, shading='nearest')
#    # make the top 0 degrees and the angles go clockwise
#    ax.set_theta_direction(-1)
#    ax.set_theta_offset(np.pi / 2.0)
#    # try to set the radial lines
#    rtick_locs = np.arange(0,25000,5000)
#    rtick_labels = ['%.1f'%r for r in rtick_locs]
#    ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
#    #
#    fig.colorbar(psm, ax=ax)
#    # Add a title
#    plt.title('Quadmesh on Polar Coordinates (matplotlib): ' + selected_field)

    print('**** done with plot_data_scatter  ***')

    return fig


params2 = [
    'Lat', 'Lon', 'Site', 'something',
    'Efficiency', 'Power', 'Displacement'
]
params = [
    'VEL', 'DBZ', 'RHOHV', 'Height',
    'Efficiency', 'Power', 'Displacement'
]

# --------- from matplotlib-dash -----
# -------- integrate from Part3: interactive Graphing and Crossfiltering ----


app.layout = html.Div([

    html.Div(
        className="app-header",
        children=[
            html.Div('HawkEdit Dashboard', className="app-header--title")
        ]
    ),
    html.Div([
        html.Div([
            dcc.Input(
               id='file-url-selector',
               type='text',
               value='/Users/brenda/data/PRECIP',
               style={'width': '30%'}
            ),
            html.Button('Open', id='open-file-folder'),
            html.Div(id='file-selection'), 
        ],
        style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Textarea(
               id='scripts',
               value='scripts',
               style={'width': '30%'}
            ),
            dcc.Input(id='input-on-submit', type='text'),
            html.Button('Run', id='run-script'),
            html.Div(id='container-button-basic',
               children='Enter a value and press submit'),
        ],
        style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Textarea(
               id='height-selector',
               value='height',
               style={'width': '30%'}
            )
        ],
        style={'width': '25%', 'display': 'inline-block'}),
    ], style={
        'padding': '10px 5px'
    }),
   
    html.Div([
        html.Div(id='time-line'),
    ], style={
        'padding': '10px 5px'
    }),


    html.Div([
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-1-1',
            ),
            html.Img(
               id='polar-1-1',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-1-2',
            ),
            html.Img(
               id='polar-1-2',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-1-3',
            ),
            html.Img(
               id='polar-1-3',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
    ], style={
        'padding': '10px 5px'
    }),
    html.Div([
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-2-1',
            ),
            html.Img(
               id='polar-2-1',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-2-2',
            ),
            html.Img(
               id='polar-2-2',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                field_names_8,
                'VEL',
                id='field-selection-2-3',
            ),
            dcc.Graph(
               id='polar-2-3',
               figure=go.Figure(data=
                  go.Scatterpolar(
                      r = [0.5,1,2,2.5,3,4],
                      theta = [35,70,120,155,205,240],
                      mode = 'markers',
                  )),
               style={'height': '150%'}
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dash_table.DataTable(
            id='system-info',
            columns=(
                [{'id': 'system-info2', 'name': 'System Info'}] +
                [{'id': p, 'name': p} for p in params2]
            ),
            data=[
                dict(Model=i, **{param: 0 for param in params2})
                for i in range(1, 5)
            ],
            editable=False
        ),        
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([

        html.Div([
            dcc.Dropdown(
                field_names_8,
                field_names_8[0],
                id='field-selector',
                multi=True,
                style={'width': '50%'},
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                ['unfold','delete','etc'],
                'unfold',
                id='actions-selector',
                style={'width': '50%'},
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'}),
        html.Div([
            dcc.Textarea(
               id='number-of-rays-display-text',
               value='number of rays to display',
               style={'width': '10%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                ['1','2','3','4'],
                '1',
                id='number-of-rays-selector',
                style={'width': '20%', 'display': 'inline-block'},
            ),
        ],
        style={'width': '100%', 'display': 'inline-block'}),

        dash_table.DataTable(
            id='spreadsheet',
            columns=(
                [{'id': 'spreadsheet-range-column', 'name': 'Range'}] +
                [{'id': 'spreadsheet-field-1', 'name': 'VEL'}]
            ),
            data=[
                dict(Model=i, **{param: 0 for param in params})
                for i in range(1, 5)
            ],
            editable=True
        ),
    ], style={
        'padding': '10px 5px'
    }),
])

#<!-- 
#        dag.AgGrid(
#            id="grid-scroll-to",
#            columnDefs=columnDefs,
#            rowData=df.to_dict("records"),
#            columnSize="sizeToFit",
#            defaultColDef={"minWidth": 150},
#            dashGridOptions={"animateRows": False}
#        ),
#-->

@app.callback(
    Output(component_id='polar-1-1', component_property='src'),
    Input('field-selection-1-1', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    fig_bar_matplotlib2 = plot_data_polly(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return fig_bar_matplotlib2

@app.callback(
    Output(component_id='polar-1-2', component_property='src'),
    Input('field-selection-1-2', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    fig_bar_matplotlib2 = plot_data_polly(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return fig_bar_matplotlib2

@app.callback(
    Output(component_id='polar-1-3', component_property='src'),
    Input('field-selection-1-3', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    fig_bar_matplotlib2 = plot_data_polly(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return fig_bar_matplotlib2

@app.callback(
    Output(component_id='polar-2-1', component_property='src'),
    Input('field-selection-2-1', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    fig_bar_matplotlib2 = plot_data_polly(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return fig_bar_matplotlib2

@app.callback(
    Output(component_id='polar-2-2', component_property='src'),
    Input('field-selection-2-2', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green")
    fig_bar_matplotlib2 = plot_data_polly(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return fig_bar_matplotlib2

@app.callback(
    Output(component_id='polar-2-3', component_property='figure'),
    Input('field-selection-2-3', 'value'),
)
def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
    print("inside green-blue")
    scatter_plot = plot_data_scatter(selected_field, max_range, beta, field, is_mdv)
    print('**** done with polar 2 ***')
    return scatter_plot

# callbacks for click centering spreadsheet data
@callback(
    Output('spreadsheet', 'columns'),
    Output('spreadsheet', 'data'),
    Input('polar-2-3', 'clickData'),
    Input('field-selection-2-3', 'value'),
)
def display_click_data(clickData, selected_field):
    print("clickData r,theta: ", clickData['points'][0]['r'], ",", clickData['points'][0]['theta'])
    theta = clickData['points'][0]['theta']
    range = clickData['points'][0]['r']
# country_name = hoverData['points'][0]['customdata']
    params=[selected_field]
    print("params: ", params)
    print("selected_field: ", selected_field)

    field_az = selected_field + " " + str(np.round(theta, decimals=2))

# changes must be in this format ...
#            columns=(
#                [{'id': 'spreadsheet2', 'name': 'Range'}] +
#                [{'id': p, 'name': p} for p in params]
#            ),

    columns = (
       [{'id': 'spreadsheet-range-column', 'name': 'Range'}] +
       [{'id': 'spreadsheet-field-1-column', 'name': field_az}]
    )

# changes must be in this format ... 
#            data=[
#                dict(Model=i, **{param: 0 for param in params})
#                for i in range(1, 5)
#            ],  
    ranges = get_ranges()
    # array[start:stop:step]
    corresponding_range_index = find_nearest_index(ranges, range)
    print("corresponding_range_index: ", corresponding_range_index)
    range_range = np.arange(corresponding_range_index-5, corresponding_range_index+5)
    print("range_range: ", range_range)
    print("ranges: ", ranges[:5])
    range_start = ranges[0] 
    range_stop = 100
    range_step = ranges[1] - ranges[0] 
    print("before get_field_data")
    field_data = get_field_data_blah(selected_field, theta)
    print("after get_field_data")
#    print("field_data: ", field_data[:5])
    data = [{ 'spreadsheet-range-column': ranges[i],    # (i*range_step)+range_start,
       'spreadsheet-field-1-column': field_data[i],
       } for i in range_range # np.arange(corresponding_range_index-5, corresponding_range_index+5)

#   corresponding_range_index-5, ... corresponding_range_index+5

    ]
    return columns, data


@callback( 
    Output(component_id='time-line', component_property='children'),  
    Output('file-selection', 'children'),
    Input('open-file-folder', 'n_clicks'),
    State('file-url-selector', 'value'),
    prevent_initial_call=True
)              
def open_file(n_clicks, path):   # really, this is setup the time slider; not open_file
    file_list = [] # {}
    if os.path.isdir(path):
       # open folder and get list of files
       print("path is a folder")
       i = 0
       with os.scandir(path) as it:
           for entry in it:
               if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith('.nc'):
                   print(entry.name)
                   # marks dictionary index for slider must be an integer!
                   file_list.append(entry.name) 
                   i += 1
    elif os.path.isfile(path):
       # /Users/brenda/data/PRECIP/SEA20220702_005700_ppi.nc
       datatree = xd.io.open_cfradial1_datatree(path)
       field_names_8 = get_field_names(datatree)
    else:
       print("not a file or folder")   

    file_list.sort()

    # Run the command and capture the output
    #result = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    # needs absolute path to lrose, or lrose must be in the environment variable PATH
#    try:    
#       result = subprocess.run(["~/lrose/bin/RadxPrint", "-h"], capture_output=True, check=True,
#          text=True)
#       if result.returncode != 0:
#          print("result: ", result.returncode)
#       # Print the output
#       print(result.stdout)
#    except CalledProcessError as e:
#       print("Errors: ", e)
           
#    return f'Output: {value}' 
#   0, 10, step=None, marks={ 0: '0°F', 3: '3°F', 5: '5°F', 7.65: '7.65°F', 10: '10°F' }, value=5
    return [dcc.Slider(0, len(file_list), step=1, 
       marks={i: ''  for i in range(len(file_list))},
       value=0, id='time-line-selector',
       #marks=marks,
       tooltip={
          "always_visible": False,
          "placement": "top"},
       ),
       dcc.Dropdown(
          file_list,
          'file',
          id='file-selection',
       )]

#    return dcc.Slider(-5, 10, 1, value=-3, id='time-line-selector')
#    return 'The input value was "{}" and the button has been clicked {} times'.format(
#        value,
#        n_clicks
#    )

@callback(
    Output('container-button-basic', 'children'),
    Input('run-script', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)
def update_output(n_clicks, value):
    # Run the command and capture the output
    #result = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    # needs absolute path to lrose, or lrose must be in the environment variable PATH
    try:
       result = subprocess.run(["~/lrose/bin/RadxPrint", "-h"], capture_output=True, check=True,
          text=True)
       if result.returncode != 0:
          print("result: ", result.returncode)
       # Print the output
       print(result.stdout)
    except CalledProcessError as e:
       print("Errors: ", e)

    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )


# this depends on new url, directory, file selected
#@callback(
#    Output(),
#    Input('time-line-selector', 'value'),
#)
#def time_line_selection_changed():


#
#@callback(
#    Output("grid-scroll-to", "scrollTo"),
#    Input("row-index-scroll-to", "value"),
#    Input("column-scroll-to", "value"),
#    Input("row-position-scroll-to", "value"),
#    Input("column-position-scroll-to", "value"),
#    State("grid-scroll-to", "scrollTo"),
#)
#def scroll_to_row_and_col(row_index, column, row_position, column_position, scroll_to):
#    if not scroll_to:
#        scroll_to = {}
#    scroll_to["rowIndex"] = row_index
#    scroll_to["column"] = column
#    scroll_to["rowPosition"] = row_position
#    scroll_to["columnPosition"] = column_position
#    return scroll_to
#


#@callback(
#    Output('spreadsheet', 'data'),
#    Input('polar-2-3', 'clickData'),
#    Input('field-selection-2-3', 'value'),
#)
#def display_click_data_rows(clickData, selected_field):
#    print("clickData r,theta: ", clickData['points'][0]['r'], ",", clickData['points'][0]['theta'])
# country_name = hoverData['points'][0]['customdata']
#    params=[selected_field]
#    print("params: ", params)

# changes must be in this format ... 
#            data=[
#                dict(Model=i, **{param: 0 for param in params})
#                for i in range(1, 5)
#            ],  
#    range_start = 10
#    range_stop = 100
#    range_step = 10
#    print("before get_field_data")
#    field_data = get_field_data(selected_field, theta)
#    print("after get_field_data")
#    # print("field_data: ", field_data[:5])
#    return [{ 'spreadsheet-range-column': (i*range_step)+range_start,
#       'spreadsheet-field-1-column': i*3,
#       } for i in range(5)
#    ]

#
## Create interactivity between dropdown component and graph
#@app.callback(
#    Output(component_id='bar-graph-matplotlib', component_property='src'),
#    Output('bar-graph-plotly', 'figure'),
#    Output('grid', 'defaultColDef'),
#    Input('category', 'value'),
#)
#
#
## Now, integrate the real data into this function, then into holoviews wrapper of quadmesh polar
#def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
#    print('inside orange')
#    # uses global datatree ...
#    sweep = datatree['/sweep_8']
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
#        max_range_index = 100
## the azimuth need to be sorted into ascending order
#        theta = np.linspace(0, 2 * np.pi, 360) # azvals  
#        # r = np.linspace(0,1, max_range_index) 
#        r = rvals[:max_range_index]
#        R, Theta = np.meshgrid(r, theta)
#        fieldvar = sweep[selected_field] # sweep[field]
#        #                              (nrows, ncolumns)
#        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
#        z = fieldvar.data[:,:max_range_index]
#        scale_factor = 360/len(azvals) # or min_distance_between_rays * 360???
#        z_bin_sort = np.full((360,max_range_index), fill_value=-3200)
#        for i in range(0,360):
#            raw_az = azvals[i]
#            new_z_index = int(raw_az/scale_factor)
#            if (new_z_index >= 360):
#                new_z_index -= 360
#            z_bin_sort[new_z_index] = z[i]
#        Z = np.nan_to_num(z_bin_sort, nan=-32)
#        # Z = np.sin(R) * np.cos(Theta)
#    # return  hv.QuadMesh((R, Theta, Z)).options(projection='polar', cmap='seismic',) 
#    #  Create a polar plot
#    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
#    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
#    # Plot the quadmesh
#    #            (X(column), Y(row), Z(row,column))
#    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
#    psm = ax.pcolormesh(Theta, R, Z, 
#        cmap=cmap,
#        vmin=edges_norm[0], vmax=edges_norm[-1],
#        # norm=norm,
#        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)), 
#        rasterized=True, shading='nearest')
#    # make the top 0 degrees and the angles go clockwise
#    ax.set_theta_direction(-1)
#    ax.set_theta_offset(np.pi / 2.0)
#    # try to set the radial lines
#    rtick_locs = np.arange(0,25000,5000)
#    rtick_labels = ['%.1f'%r for r in rtick_locs]
#    ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
#    # 
#    fig.colorbar(psm, ax=ax)
#    # Add a title
#    plt.title('Quadmesh on Polar Coordinates (matplotlib): ' + selected_field)
#
#
#    # Save it to a temporary buffer.
#    buf = BytesIO()
#    fig.savefig(buf, format="png")
#    # Embed the result in the html output.
#    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
#    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
#
#    print('**** done with polar ***')
#
#    # Build the Plotly figure
#    selected_yaxis = 'Number of Solar Plants'
#    fig_bar_plotly = px.bar(df, x='State', y=selected_yaxis).update_xaxes(tickangle=330)
#
#    my_cellStyle = {
#        "styleConditions": [
#            {
#                "condition": f"params.colDef.field == '{selected_yaxis}'",
#                "style": {"backgroundColor": "#d3d3d3"},
#            },
#            {   "condition": f"params.colDef.field != '{selected_yaxis}'",
#                "style": {"color": "black"}
#            },
#        ]
#    }
#
#    return fig_bar_matplotlib, fig_bar_plotly, {'cellStyle': my_cellStyle}
#
#
## working here ...
## syntax is Input('<widget-id>', 'value')
## arguments to python function are in order of Input from top to bottom
# use this when changing sweep height, will update all images
##@app.callback(
##    Output(component_id='bar-graph-matplotlib',  component_property='src'),
##    Output(component_id='bar-graph-matplotlib2', component_property='src'),
##    Input('sweep_height', 'value'),
##    Input('category', 'value'),
##    Input('category2', 'value'),
##    )
##def update_graphs_new_sweep_selected(sweep_height_value, selected_field_name='SW',
##    selected_field_name2='SW'):
##
##    print('inside yellow')
##    sweep_node_name = "/sweep_".append(sweep_height_value)
##    print('sweep_node_name = ', sweep_node_name)
##    image_matplotlib_window1 = plot_data_polly(selected_field_name, max_range=100, beta=sweep_node_name, 
##        field='ZDR', is_mdv=True) 
##    image_matplotlib_window2 = plot_data_polly(selected_field_name2, max_range=100, beta=sweep_node_name, 
##        field='ZDR', is_mdv=True) 
##    return image_matplotlib_window1, image_matplotlib_window2  
#
## end of working here ...
#
#
##@app.callback(
##    Output(component_id='bar-graph-matplotlib2', component_property='src'),
##    Output(component_id='bar-graph-matplotlib', component_property='src'),
##    Input('sweep_height', 'value'),
##)
### immediately following function is associated with the callback
##def change_sweep(selected_sweep):
##    
##    plot_data(selected_field, max_range=100, beta="sweep_".append(selected_sweep), field='ZDR', is_mdv=True)
##    return fig_bar_matplotlib2
#
#
## use for cartesian data
## ds = dataset
## beta is the sweep number 
#def waves_image_mdv(max_range, beta, field, ds):
#    # uses dataset sturcture  ...
#    # Q: does xradar read cartesian files?
#    # ValueError: cannot rename 'sweep_number' because it is not a variable or coordinate in this dataset
#    # for cartesian data, it is about the z-level instead of the sweep
#    height_index = 3
#    if True:
#        x0 = ds
#        # sweep_dataarray = xr.DataArray(data_2d, coords=[sweep_azimuths, sweep_range], dims=["az", "range"])
#        # R, Theta = np.meshgrid(sweep_range, sweep_azimuths)
#        # ax.pcolormesh(Theta, R, data_2d, cmap='viridis')
#        # Add a title
#        # plt.title('Quadmesh on Polar Coordinates true data')
## --- code for data tree possibly ---
#        max_range_index = 100
## the azimuth need to be sorted into ascending order
#        X0, Y0 = np.meshgrid(ds.x0.data, ds.y0.data)
#        fieldvar = ds.ZDR[0,height_index,:,:].data
#        #                              (nrows, ncolumns)
#        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
#        # z = fieldvar.data[:,:max_range_index]
#        Z = np.nan_to_num(fieldvar, nan=-32)
#    # return  hv.QuadMesh((R, Theta, Z)).options(projection='polar', cmap='seismic',)
#    #  Create a polar plot
#    fig, ax = plt.subplots(subplot_kw=dict())
#    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
#    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
#    # Plot the quadmesh
#    #            (X(column), Y(row), Z(row,column))
#    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
#    psm = ax.pcolormesh(X0, Y0, Z,
#        cmap=cmap,
#        vmin=edges_norm[0], vmax=edges_norm[-1],
#        # norm=norm,
#        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
#        rasterized=True, shading='nearest')
#    # try to set the radial lines
#    # rtick_locs = np.arange(0,25000,5000)
#    # rtick_labels = ['%.1f'%r for r in rtick_locs]
#    # ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
#    #
#    fig.colorbar(psm, ax=ax)
#    # Add a title
#    plt.title('Quadmesh on Cartesian Coordinates HV: ' + field)
#    # Show the plot
#    #plt.show()
#    # fig
#    return fig
#
##-----
#
#if __name__ == '__main__':
#    app.run_server(debug=False, port=8002)
#

if __name__ == '__main__':
    app.run(debug=True)

