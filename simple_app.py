


# import numpy as np
import holoviews as hv
import pandas as pd
import panel as pn
# import hvplot.pandas
import xarray as xr
import xradar as xd
#import cmweather
import matplotlib.pyplot as plt
#import pyproj
#import cartopy
# import hvplot.xarray


import numpy as np

from holoviews import opts
hv.extension('matplotlib')  # plotly, bokeh, matplotlib


is_cfradial = True

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


color_scale_base = "/Users/brenda/git/lrose-displays/lrose-displays-master/color_scales"
file_name = "zdr_color"
color_scale_file = color_scale_base + "/" + file_name
color_names = []
edges = []
# read the color map file
f = open(color_scale_file, "r")
for line in f.readlines():
    if line[0] != '#':
        print(line)
        x = line.split()
        print(x)
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
    norm.vmin=-32656
    norm.vmax=32656
    (zcmap, znorm) = colors.from_levels_and_colors(edges, color_scale_hex, extend='neither')
except ValueError as err:
    print("something went wrong first: ", err)

# NEXRAD
# TODO: make a default datatree structure
if is_cfradial:
    dirname = "/Users/brenda/data/for_mica/nexrad/output/20240510"
    filename = "cfrad.20240510_010615.273_to_20240510_011309.471_KBBX_SUR.nc"
    localfilename = dirname + "/" + filename
    datatree = xd.io.open_cfradial1_datatree(localfilename)
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

file_selector_widget = pn.widgets.FileSelector('~/data')

card = pn.Card(file_selector_widget, title='Step 1. Choose Data File', styles={'background': 'WhiteSmoke'})

# x = pn.widgets.IntSlider(name='sweep', start=0, end=get_sweeps(datatree)-1)
background = pn.widgets.ColorPicker(name='Background', value='lightgray')
# field_names_widget = pn.widgets.Select(name="field", options=get_field_names(datatree))
open_file_widget = pn.widgets.Button(name="open/read file?", button_type='primary')


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
        return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(max_range=[100,200,300], beta=['/sweep_0'], field=['ZDR', 'DBZH', 'RHOHV'])
    else:
        if is_cfradial:
            datatree = xd.io.open_cfradial1_datatree(file_name[0])
            pn.state.log(f'after cfradial datatree = {datatree.groups}  ')
            # filter the groups by keyword sweep
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


def show_status_open_file(dummy=1):
    return f'reading ...'

def styles(background):
    return {'background-color': background, 'padding': '0 10px'}

def get_plot(field, datatree):
    sweep = datatree['/sweep_0']
    rvals = sweep.range
    azvals = sweep.azimuth
    return hv.Image(sweep.ZDR)
#     xs, ys = np.meshgrid(rvals, azvals)
#     return hv.Image(sweep[field], xs, ys)
    

#------
# from DynamicMap tutorial ...

xvals = np.linspace(-4, 0, 100)
yvals = np.linspace(4, 0, 100)
xs, ys = np.meshgrid(xvals, yvals)
# 
# HERE datatree must not be sent!!! 
#
def waves_image_old(max_range, beta, field):  # , dtree=None):
    if hasattr(datatree, 'groups'):
        if (len(datatree.groups) > 0):
            ls = np.linspace(0, 10, 200)
            xx, yy = np.meshgrid(ls, ls)
            bounds=(-1,-1,1,1)   # Coordinate system: (left, bottom, right, top)
            return hv.Image(np.sin(xx)*np.cos(yy), bounds=bounds)
            # return hv.Image(datatree['/sweep_0'].ZDR)
        else:
            return hv.Image(np.sin(((ys/max_range)**max_range+beta)*xs))
    else:
        return hv.Image(np.sin(((ys/max_range)**max_range+beta)*xs))

def waves_image(max_range, beta, field):
    if (len(beta)):
        # uses global datatree ...
        sweep_name = beta
        sweep = datatree[sweep_name] # ['/sweep_8']
        rvals = sweep.range
        azvals = sweep.azimuth
        max_range = map_range_to_index(max_range, rvals,
            sweep.ray_gate_spacing.data[0],
            sweep.ray_start_range.data[0])  # 100 # 300
        # theta = azvals
        # the azimuth need to be sorted into ascending order
        # theta = azvals #  np.linspace(0, 2 * np.pi, 720) # azvals
        theta = np.linspace(0, 2 * np.pi, 360)
        r = rvals[:max_range]
        R, Theta = np.meshgrid(r, theta)
        fieldvar = sweep[field]
        ##                       shape = (|az|, |range|)
        #pn.state.log(f'fieldvar.shape = {fieldvar.shape}  ')
        ##                              (nrows, ncolumns)
        ## z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        #z2 = fieldvar.data[:,:max_range] 
        #pn.state.log(f'z2.shape = {z2.shape}  ')
        ##z = fieldvar.data 


        z = fieldvar.data[:,:max_range]
        scale_factor = 306/len(azvals) # or min_distance_between_rays * 360???
        z_bin_sort = np.zeros((360,max_range))
        for i in range(0,360):
            raw_az = azvals[i]
            new_z_index = int(raw_az/scale_factor)
            if (new_z_index >= 360):
                new_z_index -= 360
            z_bin_sort[new_z_index] = z[i]



        Z = np.nan_to_num(z_bin_sort, nan=-32)
        # get the color map
        cmap = colors.ListedColormap(color_scale_hex)
        # add options using the Options Builder
        try:
            img = hv.QuadMesh((Theta, R, Z)).opts(opts.QuadMesh(cmap=cmap, projection='polar',
                colorbar = True,
                #norm=colors.BoundaryNorm(edges, ncolors=len(edges)), # causes an error
                #norm=norm
                ))
        except ValueError as err:
            print("something went wrong: ", err)
    else:
        # use test data ..
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 1, 100)
        R, Theta = np.meshgrid(r, theta)
        Z = np.sin(R) * np.cos(Theta)
        img = hv.QuadMesh((Theta, R, Z)).opts(opts.QuadMesh(cmap='jet', projection='polar'))
    return img


# Now, integrate the real data into this function, then into holoviews wrapper of quadmesh polar
def waves_image_new(max_range, beta, field):
    # uses global datatree ...
    pn.state.log(f'beta =  ... ')
    sweep = datatree['/sweep_8']
    rvals = sweep.range
    azvals = sweep.azimuth
    #return hv.Image(sweep.ZDR)
    # Generate data in polar coordinates
    test_data = False # True
    if test_data:
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 1, 100)
        R, Theta = np.meshgrid(r, theta)
        Z = np.sin(R) * np.cos(Theta)
    else:
        # construct an xarray.DataArray ...
        #sweep_number = 8
        #sweep_start_ray_index = int(ds.sweep_start_ray_index.data[sweep_number])
        #sweep_end_ray_index = int(ds.sweep_end_ray_index.data[sweep_number])
        #sweep_azimuths = ds.azimuth.data[sweep_start_ray_index:sweep_end_ray_index+1]
        #n_gates = int(ds.ray_n_gates.data[sweep_start_ray_index])
        #gate_spacing = int(ds.ray_gate_spacing.data[sweep_start_ray_index])
        #start_range = int(ds.ray_start_range.data[sweep_start_ray_index])
        #end_gate = start_range+(n_gates*gate_spacing)
        #sweep_range = np.linspace(start_range, end_gate, n_gates, endpoint=False)
        ## ray_n_gates(time) float64
        ## ray_gate_spacing(time)
        ## VEL(n_points)
        #npoints_start_ray_n = int(ds.ray_start_index.data[sweep_start_ray_index])
        #npoints_end_ray_n = int(ds.ray_start_index.data[sweep_end_ray_index] + ds.ray_n_gates.data[sweep_start_ray_index])
        #ray_data = ds.RHO.data[npoints_start_ray_n:npoints_end_ray_n]
        # data_2d = np.reshape(ray_data, (len(sweep_azimuths), n_gates))
        # sweep_dataarray = xr.DataArray(data_2d, coords=[sweep_azimuths, sweep_range], dims=["az", "range"])
        # R, Theta = np.meshgrid(sweep_range, sweep_azimuths)
        # ax.pcolormesh(Theta, R, data_2d, cmap='viridis')
        # # Add a title
        # plt.title('Quadmesh on Polar Coordinates true data')
        max_range = 100
# the azimuth need to be sorted into ascending order
        theta = np.linspace(0, 2 * np.pi, 360) # azvals  
        # r = np.linspace(0,1, max_range) 
        r = rvals[:max_range]
        R, Theta = np.meshgrid(r, theta)
        fieldvar = sweep[field]
        #                              (nrows, ncolumns)
        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        z = fieldvar.data[:,:max_range]
        scale_factor = 306/len(azvals) # or min_distance_between_rays * 360???
        z_bin_sort = np.zeros((360,max_range))
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
    # Plot the quadmesh
    #            (X(column), Y(row), Z(row,column))
    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
    psm = ax.pcolormesh(Theta, R, Z, 
        cmap=zcmap,
        # norm=znorm,
        norm=colors.BoundaryNorm(edges, ncolors=len(edges)), 
        rasterized=True, shading='nearest')
    fig.colorbar(psm, ax=ax)
    # Add a title
    plt.title('Quadmesh on Polar Coordinates HV: ' + field)
    # Show the plot
    #plt.show()
    # fig
    return fig 

def waves_image_new1(max_range, beta, field):
    # Generate data in polar coordinates
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 1, 100)
    R, Theta = np.meshgrid(r, theta)
    Z = np.sin(R) * np.cos(Theta)
    # return  hv.QuadMesh((R, Theta, Z)).options(projection='polar', cmap='viridis',)
    #  Create a polar plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # Plot the quadmesh
    ax.pcolormesh(Theta, R, Z, cmap='viridis')
    # Add a title
    plt.title('Quadmesh on Polar Coordinates HV')
    # Show the plot
    #plt.show()
    # fig
    return fig

dmap = hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field'])

#-----

my_column = pn.Column(
    waves_image_new(1,0,'ZDR'),
    # dmap[1,2] + dmap.select(max_range=1, beta=2),
    card,
    pn.panel(pn.bind(show_selected_file, file_selector_widget), backend='matplotlib'), # , styles=pn.bind(styles, background))
)

# make this into a stand alone app
pn.template.MaterialTemplate(
    site="MICA",
    title="Getting Started App",
    main=[my_column]
).servable(); # The ; is needed in the notebook to not display the template. Its not needed in a script
