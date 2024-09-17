#!/usr/bin/env python
# coding: utf-8

# In[30]:


import holoviews as hv
import panel as pn
pn.extension()


# In[2]:


file_selector_widget = pn.widgets.FileSelector('~/data')

card = pn.Card(file_selector_widget, title='Step 1. Choose Data File', styles={'background': 'WhiteSmoke'})


# In[3]:


file_selector_widget


# In[4]:


pn.extension(design="material", sizing_mode="stretch_width")


# In[ ]:





# In[5]:


# file_selector_widget.value


# In[ ]:


#bound_card = pn.bind(
#    get_data_location, file_dir=variable_widget, window=window_widget, sigma=sigma_widget
#)


# In[6]:


#widgets = pn.Column(variable_widget, window_widget, sigma_widget, sizing_mode="fixed", width=300)
#pn.Column(widgets, bound_plot)
pn.Column(card)


# In[ ]:





# In[80]:


import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas
import xarray as xr
import xradar as xd
import cmweather
import matplotlib.pyplot as plt
import pyproj
#import cartopy
import hvplot.xarray


# In[5]:


localfilename = file_selector_widget.value
display(localfilename[0])
type(localfilename[0])
type("one")


# In[31]:


#help(xd.io)


# In[81]:


# Use this after binding file chooser to open file
# ds = xr.open_dataset(localfilename[0])
#ds = xd.io.open_cfradial1_datatree("/Users/brenda/data/cfrad.20170408_001452.962_to_20170408_002320.954_KARX_Surveillance_SUR.nc",
#                                  first_dim="auto")
  # PRECIP/SEA20220702_001310_ppi.nc") # , group="sweep_0") # , engine="nexradlevel2")
# ds = xd.io.open_cfradial1_datatree("/Users/brenda/data/cfrad.20161006_190650.891_to_20161006_191339.679_KAMX_SUR.nc")
#                                  group="sweep_0", engine="nexradlevel2" )
#ds = xd.io.open_dataset("/Users/brenda/data/cfrad.20170408_001452.962_to_20170408_002320.954_KARX_SUR.nc")

#ds = xr.open_dataset(filename, group="sweep_0", engine="nexradlevel2")

# help(xd.open_datatree)
# NEXRAD
path = "/Users/brenda/data/for_mica/nexrad/"
filename = "KBBX20240510_010615_V06"
datatree = xd.io.backends.nexrad_level2.open_nexradlevel2_datatree(path+filename)
display(datatree)

# there is an error in xradar datatree, it wants n_points, which it doesn't find.
# ncdump of the file has n_points, but just reading cfradial1 using datatree,
# there is no n_points.
#path = "/Users/brenda/data/for_mica/nexrad/output/20240510/"
#filename = "cfrad.20240510_010615.273_to_20240510_011309.471_KBBX_SUR.nc"
#ds = xd.io.backends.cfradial1.open_cfradial1_datatree(path+filename, sweep=0)
#display(ds)


# In[38]:


def rms(signal):
    return np.sqrt(np.mean(signal**2))


# In[84]:


for group in list(datatree.groups):
    print(group)

# data tree is a tuple, an ordered list of elements, the first element is the root '/'

display(type(datatree.groups))
root = datatree.groups
display(root[0])
display(root[2])
display(len(root))
sweep = root[2]
display(type(sweep))
sweeps = datatree.children
display(type(sweeps))

print("is hollow? ", datatree.is_hollow)
# [node.data_vars for node in datatree.leaves]
# node.children gets the name of the node as a string
# node.subtree generates an iterable over the nodes of the tree in depth-first order
for node in datatree.subtree:
    print(node.name, " path: ", node.path)

# get field variables from a datatree ...
sweep0 = datatree["/sweep_0"]
display(type(sweep0))
#display(sweep0.data_vars)
print("dims: ", sweep0.dims)
print("data_vars: ", sweep0.data_vars)
sweep0_data_vars = sweep0.data_vars
fieldnames = []
for k, v in sweep0_data_vars.items():
    print(k, len(v.dims))
    if len(v.dims) >= 2:
        fieldnames.append(k)


display("fields names: ", fieldnames)


# In[86]:


#list(ds.data_vars)
# get field variables from a dataset
def get_field_names(dataset):
    field_names = []
    ds = dataset
    for v in list(ds.data_vars):
        the_var = ds.data_vars[v]
        print(type(the_var))
        dim = len(ds.data_vars[v].shape)
        print(v, dim, ds.data_vars[v].shape) # , len(the_var))
        if dim == 2:
            #print(v)
            field_names.append(v)

#field_names = ['SDP', 'SQI2', 'DBZ2', 'FDP', 'WIDTH2', 'KDP', 'KDP2', 'DBT2', 'RHOHV2', 'SNR16', 'CC', 'ZDR2', 'HID', 'CV', 'PHIDP2', 'UNKNOWN_82', 'VEL2', 'DZQC']
#print(field_names)
#field_names_widget = pn.widgets.Select(name="field", options=list(field_names))
#pn.Column(field_names_widget)


# In[6]:


#def get_data_set(variable):
#    ds = xr.open_dataset(variable, engine='cfradial1')
#    return ds

def get_fields(variable=[]):  # ="*.nc"):
    print(variable)
    field_names = []
    if (variable):
        ds = xr.open_dataset(variable) # , engine='cfradial1')  # "/home/jovyan/notebooks/data-access/ARPA_Lombardia.20240522.151546_dealiased.nc") # , group="sweep_0") # , engine="nexradlevel2")
        for v in list(ds.data_vars):
            # print(v)
            dim = len(ds.data_vars[v].shape)
            if dim == 2:
                print(v)
                field_names.append(v)
    field_names_widget = pn.widgets.Select(name="field", options=list(field_names))
    return field_names_widget


# In[ ]:


#test_names = get_fields("/home/jovyan/notebooks/data-access/ARPA_Lombardia.20240522.151546_dealiased.nc")
#test_names


# In[8]:


#bound_plot = pn.bind(
#    get_data_set, variable=file_selector_widget
#)

#bound_plot2 = pn.bind(
#    get_fields, ds=get_data_set
#)
#wd = get_fields("/Users/brenda/data/PRECIP/SEA20220702_001310_ppi.nc")
#wd


# In[9]:


#
#pst = get_fields(dst)
#pst
bound_get_fields = pn.bind(get_fields, variable=file_selector_widget)
pn.Column(card, bound_get_fields)

#pn.Column(
#    #"Which calculation would you like to perform?",
#    card,
#    ,
#    get_fields
#).servable()


# In[12]:


# make this into a stand alone app
pn.template.MaterialTemplate(
    site="Panel",
    title="Getting Started App",
    #sidebar=[field_names_widget],
    #main=[card]
    # main=[field_names_widget]
    main=[card, bound_get_fields]
).servable(); # The ; is needed in the notebook to not display the template. Its not needed in a script


# In[5]:


#help(pn.servable)
#help(pn.viewable)

#def transform_data(variable, window, sigma):
#    """Calculates the rolling average and identifies outliers"""
#    avg = data[variable].rolling(window=window).mean()
#    residual = data[variable] - avg
#    std = residual.rolling(window=window).std()
#    outliers = np.abs(residual) > std * sigma
#    return avg, avg[outliers]


#def get_plot(variable="Temperature", window=30, sigma=10):
#    """Plots the rolling average and the outliers"""
#    #avg, highlight = transform_data(variable, window, sigma)
#    avg = [1,3,4,5,6]
#    return avg.hvplot(
#        height=300, legend=False, color=PRIMARY_COLOR
#    ) #  * highlight.hvplot.scatter(color=SECONDARY_COLOR, padding=0.1, legend=False)


# In[23]:


#get_plot(variable='Temperature', window=20, sigma=10)


# In[ ]:


# HERE ...
#variable_widget = pn.widgets.Select(name="variable", value="Temperature", options=list(data.columns))
#window_widget = pn.widgets.IntSlider(name="window", value=30, start=1, end=60)
#sigma_widget = pn.widgets.IntSlider(name="sigma", value=10, start=0, end=20)


# In[ ]:


#bound_plot = pn.bind(
#    get_plot, variable=variable_widget, window=window_widget, sigma=sigma_widget
#)


# In[ ]:


#widgets = pn.Column(variable_widget, window_widget, sigma_widget, sizing_mode="fixed", width=300)
#pn.Column(widgets, bound_plot)


# In[ ]:


#pn.template.MaterialTemplate(
#    site="Panel",
#    title="Getting Started App",
#    sidebar=[variable_widget, window_widget, sigma_widget],
#    main=[bound_plot],
#).servable(); # The ; is needed in the notebook to not display the template. Its not needed in a script


# In[ ]:


# If you prefer developing in a Python Script using an editor, you can copy the code into a file app.py and serve it.

# panel serve app.py --autoreload


# In[ ]:




