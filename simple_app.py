


import panel as pn

import numpy as np
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

# NEXRAD
path = "/Users/brenda/data/for_mica/nexrad/"
filename = "KBBX20240510_010615_V06"
datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(path+filename)





def get_field_names(dummy):
    return ['A', 'B', 'C']

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

x = pn.widgets.IntSlider(name='sweep', start=0, end=100)
background = pn.widgets.ColorPicker(name='Background', value='lightgray')
# field_names_widget = pn.widgets.Select(name="field", options=['A','E','F'])
field_names_widget = pn.widgets.Select(name="field", options=get_field_names(datatree))

def square(x):
    return f'{x} squared is {x**2}'

def show_selected_field(field, x):
    return f'selected field is {field} sweep is {x}'

def styles(background):
    return {'background-color': background, 'padding': '0 10px'}

my_column = pn.Column(
    x,
    field_names_widget,
    background,
    # pn.pane.Markdown(pn.bind(square, x), styles=pn.bind(styles, background))
    pn.pane.Markdown(pn.bind(show_selected_field, field_names_widget, x), styles=pn.bind(styles, background))
)

# make this into a stand alone app
pn.template.MaterialTemplate(
    site="Panel",
    title="Getting Started App",
    #sidebar=[field_names_widget],
    #main=[card]
    # main=[field_names_widget]
    main=[my_column]
#    main=[card, bound_get_fields]
).servable(); # The ; is needed in the notebook to not display the template. Its not needed in a script
