from dash import Dash, dash_table, html, dcc, Input, Output, callback
import plotly.express as px
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

# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

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
    html.Div([
        html.Div([
            dcc.Textarea(
               value='HawkEdit Dashboard',
               style={'width': '30%'}
            )
        ],
        style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Textarea(
               id='file-url-selector',
               value='select file/url',
               style={'width': '30%'}
            )
        ],
        style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Textarea(
               id='scripts',
               value='scripts',
               style={'width': '30%'}
            )
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
        html.Div([
            dcc.Slider(-5, 10, 1, value=-3,
               id='time-line-selector',
            )
        ]),
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
            html.Img(
               id='polar-2-3',
               src='https://loremflickr.com/1280/720',
               style={'width': '100%'}
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
                df['Indicator Name'].unique(),
                'Fertility rate, total (births per woman)',
                id='field-selector',
                style={'width': '30%'},
            ),
        ],
        style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Fertility rate, total (births per woman)',
                id='actions-selector',
                style={'width': '30%'},
            ),
        ],
        style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Fertility rate, total (births per woman)',
                id='number-of-rays-selector',
                style={'width': '30%'},
            ),
        ],
        style={'width': '20%', 'display': 'inline-block'}),

        dash_table.DataTable(
            id='spreadsheet',
            columns=(
                [{'id': 'spreadsheet2', 'name': 'Range'}] +
                [{'id': p, 'name': p} for p in params]
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

    html.Div([

        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Fertility rate, total (births per woman)',
                id='crossfilter-xaxis-column',
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-xaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='crossfilter-yaxis-column'
            ),
            dcc.RadioItems(
                ['Linear', 'Log'],
                'Linear',
                id='crossfilter-yaxis-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        df['Year'].min(),
        df['Year'].max(),
        step=None,
        id='crossfilter-year--slider',
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input('crossfilter-year--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
            )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='Year', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'))
def update_x_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-yaxis-type', 'value'))
def update_y_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run(debug=True)



#app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#app.layout = html.Div([
#    html.H1("Interactive Matplotlib with Dash", className='mb-2', style={'textAlign':'center'}),
#
#    dbc.Row([
#        dbc.Col([
#            dcc.Dropdown(
#                id='sweep_height',
#                value='0',
#                clearable=False,
#                options=sweep_names) 
#        ], width=4),
#    ]),
#
#    dbc.Row([
#        dbc.Col([
#            dcc.Dropdown(
#                id='category',
#                value='ZDR',
#                clearable=False,
#                options=field_names_8) 
#        ], width=4),
#        dbc.Col([
#            html.Img(id='bar-graph-matplotlib')
#        ], width=12)
#    ]),
#
#    dbc.Row([
#        dbc.Col([
#            dcc.Dropdown(
#                id='category2',
#                value='VEL',
#                clearable=False,
#                options=field_names_8)
#        ], width=4),
#        dbc.Col([
#            html.Img(id='bar-graph-matplotlib2')
#        ], width=12)
#    ]),
#    
#    dbc.Row([
#        dbc.Col([
#            dcc.Graph(id='bar-graph-plotly', figure={})
#        ], width=12, md=6),
#        dbc.Col([
#            dag.AgGrid(
#                id='grid',
#                rowData=df.to_dict("records"),
#                columnDefs=[{"field": i} for i in df.columns],
#                columnSize="sizeToFit",
#            )
#        ], width=12, md=6),
#    ], className='mt-4'),
#
#])
#
## Create interactivity between dropdown component and graph
#
#
##@app.callback(
##    Output(component_id='bar-graph-matplotlib2', component_property='src'),
##    Output(component_id='bar-graph-matplotlib', component_property='src'),
##    Input('sweep_height', 'value'),
##)
#
#@app.callback(
#    Output(component_id='bar-graph-matplotlib2', component_property='src'),
#    Input('category2', 'value'),
#)
#
##def plot_data_orig(selected_yaxis):
##
##    # Build the matplotlib figure
##    fig = plt.figure(figsize=(14, 5))
##    plt.bar(df['State'], df[selected_yaxis])
##    plt.ylabel(selected_yaxis)
##    plt.xticks(rotation=30)
##
##    # Save it to a temporary buffer.
##    buf = BytesIO()
##    fig.savefig(buf, format="png")
##    # Embed the result in the html output.
##    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
##    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
##
##    # Build the Plotly figure
##    fig_bar_plotly = px.bar(df, x='State', y=selected_yaxis).update_xaxes(tickangle=330)
##
##    my_cellStyle = {
##        "styleConditions": [
##            {
##                "condition": f"params.colDef.field == '{selected_yaxis}'",
##                "style": {"backgroundColor": "#d3d3d3"},
##            },
##            {   "condition": f"params.colDef.field != '{selected_yaxis}'",
##                "style": {"color": "black"}
##            },
##        ]
##    }
##
##    return fig_bar_matplotlib, fig_bar_plotly, {'cellStyle': my_cellStyle}
#
#
## -------- end of matplotlib-dash --------
#
## HERE datatree must not be sent!!! 
##
#
#def plot_data(selected_field, max_range=100, beta="sweep_x", field='ZDR', is_mdv=True):
#    print("inside green")
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
#    fig_bar_matplotlib2 = f'data:image/png;base64,{fig_data}'
#
#    print('**** done with polar 2 ***')
#
#    return fig_bar_matplotlib2
#
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
