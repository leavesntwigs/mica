


import numpy as np
import holoviews as hv
import pandas as pd
import panel as pn
import hvplot.pandas
import xarray as xr
import xradar as xd
#import cmweather
#import matplotlib.pyplot as plt
#import pyproj
#import cartopy
import hvplot.xarray

from holoviews import opts
hv.extension('matplotlib')

# NEXRAD
# TODO: make a default datatree structure
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
        return hv.DynamicMap(waves_image, kdims=['alpha', 'beta', 'field']).redim.values(alpha=[1,2,3], beta=[0.1, 1.0, 2.5], field=['DBZ', 'REF', 'RHO'])
    else:
        datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(file_name[0])
        fields = get_field_names(datatree)
        sweeps = get_sweeps(datatree)
        return hv.DynamicMap(waves_image, kdims=['alpha', 'beta', 'field']).redim.values(alpha=[1,2,3], beta=[0.1, 1.0, 2.5], field=fields) # , dtree=datatree)


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
def waves_image(alpha, beta, field):  # , dtree=None):
    if hasattr(datatree, 'groups'):
        if (len(datatree.groups) > 0):
            ls = np.linspace(0, 10, 200)
            xx, yy = np.meshgrid(ls, ls)
            bounds=(-1,-1,1,1)   # Coordinate system: (left, bottom, right, top)
            return hv.Image(np.sin(xx)*np.cos(yy), bounds=bounds)
            # return hv.Image(datatree['/sweep_0'].ZDR)
        else:
            return hv.Image(np.sin(((ys/alpha)**alpha+beta)*xs))
    else:
        return hv.Image(np.sin(((ys/alpha)**alpha+beta)*xs))

dmap = hv.DynamicMap(waves_image, kdims=['alpha', 'beta', 'field'])

#-----

my_column = pn.Column(
    # waves_image(1,0),
    # dmap[1,2] + dmap.select(alpha=1, beta=2),
    card,
    pn.panel(pn.bind(show_selected_file, file_selector_widget), backend='bokeh'), # , styles=pn.bind(styles, background))
)

# make this into a stand alone app
pn.template.MaterialTemplate(
    site="MICA",
    title="Getting Started App",
    main=[my_column]
).servable(); # The ; is needed in the notebook to not display the template. Its not needed in a script
