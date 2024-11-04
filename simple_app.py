


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


is_cfradial = False
is_mdv = True


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

color_scale_base = "/Users/brenda/git/lrose-displays/lrose-displays-master/color_scales"
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
            pn.state.log(x[2].lower)
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
        return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(max_range=[25000,30000], beta=['/sweep_8'], field=['ZDR', 'DBZH', 'RHOHV'])
    else:
        if is_mdv:
            # cartesian data set; use dataset structure
            ds_cart = xr.open_dataset(file_name[0])
            fields = get_field_names(ds_cart)
            heights = ds_cart.z0.data # get_sweeps(ds_cart)
            return hv.DynamicMap(waves_image, kdims=['max_range', 'beta', 'field']).redim.values(
                max_range=[100,200,300,400, 500, 600, 700],
                beta=heights,  
                field=fields)
        else:
            # polar data; use datatree structure
            if is_cfradial:
                datatree = xd.io.open_cfradial1_datatree(file_name[0])
                pn.state.log(f'after cfradial datatree = {datatree.groups}  ')
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
   
# this doesn't work.  Giving up on the coloring the range labels for embedded HoloViews plots 
def hook(plot, element):
    plot.handles['xaxis'].axis_label_text_color = 'red'
    plot.handles['yaxis'].axis_label_text_color = 'white'
    # legend.set_frame_on(False)
    # ...
# # rlabels = ax.get_ymajorticklabels()
# for label in rlabels:
    # label.set_color('white')

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
    if is_mdv:
        # switch to dataset and cartesian coordinates
        ds = ds_cart
        height_index = 3
        # Add a title
        # plt.title('Quadmesh on Polar Coordinates true data')
        max_range_index = 100
        X0, Y0 = np.meshgrid(ds.x0.data, ds.y0.data)
        fieldvar = ds.ZDR[0,height_index,:,:].data
        Z = np.nan_to_num(fieldvar, nan=-32)

        # get the color map
        (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
        pn.state.log(f'edges_norm = {edges_norm} ')
        pn.state.log(f'colors_norm = {colors_norm} ')
        cmap = colors.ListedColormap(colors_norm)  # (color_scale_hex)
        # add options using the Options Builder
        vmin = edges_norm[0]
        vmax = edges_norm[-1]
        img = hv.QuadMesh((X0, Y0, Z)).opts(opts.QuadMesh(cmap=cmap,
                #hooks=[hook],
                clim=(vmin, vmax),  # works if regularly spaced edges
                colorbar=True,
                # rasterized=True,
                shading='auto',
                title="",
                labelled=[],
                ))

    elif (len(beta)):
        # uses global datatree ...
        sweep_name = beta
        sweep = datatree[sweep_name] # ['/sweep_8']
        rvals = sweep.range
        azvals = sweep.azimuth
# TODO: sort out max_range: it has multiple meanings
        max_range_index = map_range_to_index(max_range, rvals,
            sweep.ray_gate_spacing.data[0],
            sweep.ray_start_range.data[0])  # 100 # 300
        # theta = azvals
        # the azimuth need to be sorted into ascending order
        # theta = azvals #  np.linspace(0, 2 * np.pi, 720) # azvals
        theta = np.linspace(0, 2 * np.pi, 360)
        r = rvals[:max_range_index]
        R, Theta = np.meshgrid(r, theta)
        fieldvar = sweep[field]
        ##                       shape = (|az|, |range|)
        #pn.state.log(f'fieldvar.shape = {fieldvar.shape}  ')
        ##                              (nrows, ncolumns)
        ## z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        #z2 = fieldvar.data[:,:max_range_index] 
        #pn.state.log(f'z2.shape = {z2.shape}  ')
        ##z = fieldvar.data 


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
        # get the color map
        (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
        pn.state.log(f'edges_norm = {edges_norm} ')
        pn.state.log(f'colors_norm = {colors_norm} ')
        cmap = colors.ListedColormap(colors_norm)  # (color_scale_hex)
        # add options using the Options Builder
        try:
            vmin = edges_norm[0]
            vmax = edges_norm[-1]
            img = hv.QuadMesh((Theta, R, Z)).opts(opts.QuadMesh(cmap=cmap,
                hooks=[hook],
                clim=(vmin, vmax),  # works if regularly spaced edges 
                # norm=znorm, # error, ValueError: Passing a Normalize instance simultaneously with vmin/vmax is not supported.  Please pass vmin/vmax directly to the norm when creating it.
                # norm=norm,
                projection='polar',
                # this is causing problems, with a vmin, vmax setting ValueError.
                # cnorm=norm, # error, does NOT accept  <matplotlib.colors.BoundaryNorm object at 0x156ae9dc0>; valid options include: '[linear, log, eq_hist]'
                colorbar=True,
                # rasterized=True, 
                shading='auto',
                title="",
                labelled=[],
                # invert_xaxis=True, start_angle=np.pi,
                ))
            # alternative option 1: get the renderer
            # fig = hv.render(img)
            #fig.axes['set_theta_direction']=-1
            # ax.set_theta_offset(np.pi / 2.0)
            # try to set the radial lines
            # rtick_locs = np.arange(0,25000,5000)
            # rtick_labels = ['%.1f'%r for r in rtick_locs]
            # fig.axes.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white") # error AttributeError: 'list' object has no attribute 'set_rgrids'
            # alternative option 2: use a hook 
            # alternative option 3:  
            # try to set the radial lines
            rtick_locs = np.arange(0,25000,5000)
            rtick_labels = ['%.1f'%r for r in rtick_locs]
            #ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
            img.opts(
                # hooks=[hook],
                backend_opts={
                    "axes.set_theta_offset":np.pi / 2.0,
                    "axes.set_theta_direction":-1,
                    "axes.yticklabels":rtick_labels,
                    "axes.yticklabel_color":"white",
                    #"axes.set_rgrids.labels":("one","two","three","four"),
            })
        except ValueError as err:
            pn.state.log(f'something went wrong: ') # , err)
    else:
        # use test data ..
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 1, 100)
        R, Theta = np.meshgrid(r, theta)
        Z = np.sin(R) * np.cos(Theta)
        img = hv.QuadMesh((Theta, R, Z)).opts(opts.QuadMesh(cmap='jet', projection='polar'))
    return img


# Now, integrate the real data into this function, then into holoviews wrapper of quadmesh polar
def waves_image_new(max_range, beta, field, is_mdv=True):
    if is_mdv:
        # switch to dataset and cartesian coordinates
        return waves_image_mdv(max_range, beta, field, ds_cart)
    # uses global datatree ...
    pn.state.log(f'beta =  ... ')
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
        fieldvar = sweep[field]
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
    plt.title('Quadmesh on Polar Coordinates HV: ' + field)
    # Show the plot
    #plt.show()
    # fig
    return fig 

# use for cartesian data
# ds = dataset
# beta is the sweep number 
def waves_image_mdv(max_range, beta, field, ds):
    # uses dataset sturcture  ...
    pn.state.log(f'beta =  ... ')
    # Q: does xradar read cartesian files?
    # ValueError: cannot rename 'sweep_number' because it is not a variable or coordinate in this dataset
    # for cartesian data, it is about the z-level instead of the sweep
    height_index = 3
    if True:
        x0 = ds
        # sweep_dataarray = xr.DataArray(data_2d, coords=[sweep_azimuths, sweep_range], dims=["az", "range"])
        # R, Theta = np.meshgrid(sweep_range, sweep_azimuths)
        # ax.pcolormesh(Theta, R, data_2d, cmap='viridis')
        # Add a title
        # plt.title('Quadmesh on Polar Coordinates true data')
# --- code for data tree possibly ---
        max_range_index = 100
# the azimuth need to be sorted into ascending order
        X0, Y0 = np.meshgrid(ds.x0.data, ds.y0.data)
        fieldvar = ds.ZDR[0,height_index,:,:].data
        #                              (nrows, ncolumns)
        #z = np.reshape(fieldvar.data, (len(azvals), len(rvals)))
        # z = fieldvar.data[:,:max_range_index]
        Z = np.nan_to_num(fieldvar, nan=-32)
    # return  hv.QuadMesh((R, Theta, Z)).options(projection='polar', cmap='seismic',)
    #  Create a polar plot
    fig, ax = plt.subplots(subplot_kw=dict())
    (edges_norm, colors_norm) = normalize_colormap(edges, color_scale_hex)
    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
    # Plot the quadmesh
    #            (X(column), Y(row), Z(row,column))
    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
    psm = ax.pcolormesh(X0, Y0, Z,
        cmap=cmap,
        vmin=edges_norm[0], vmax=edges_norm[-1],
        # norm=norm,
        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
        rasterized=True, shading='nearest')
    # try to set the radial lines
    # rtick_locs = np.arange(0,25000,5000)
    # rtick_labels = ['%.1f'%r for r in rtick_locs]
    # ax.set_rgrids(rtick_locs, rtick_labels, fontsize=16, color="white")
    #
    fig.colorbar(psm, ax=ax)
    # Add a title
    plt.title('Quadmesh on Cartesian Coordinates HV: ' + field)
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
    waves_image_new(1,0,'ZDR', is_mdv),
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
