#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import holoviews as hv
import pandas as pd
from holoviews import opts
hv.extension('bokeh')
np.random.seed(10)


# In[97]:


#help(hv.HeatMap)

# evaluate the vel function at the azimuth x
# which means, lookup the azimuth, x, in the vel table,
# using the closest index in az.
def ttfunc(obs_az, obs_vel, grid_az, nranges):
    #for i in range(len(ds_az)):
    # find x in the list of azimuths, return the index
    #closest_az = int(az[0])
    #return x
    grid_spacing = abs(grid_az[1] - grid_az[0])
    start_az = grid_az[0]
    end_az = grid_az[len(grid_az) - 1]
    #nranges = len(obs_vel[0])
    result = np.zeros((len(grid_az), nranges))
    # loop over the observations, because they could be more irregular
    # the grid azimuths will be regular
    ii = 0;
    for o_az in obs_az:
        #print(o_az)
        idx = int( (o_az - start_az) / grid_spacing )
        # print(idx, ii)
        result[idx] = obs_vel[ii]
        ii += 1
    return result
#        indexes = np.where(obs_az == g_az)
#        print(indexes)
        
#    try:
#        i = az.index(x[0])
#        return i # vel[i]
#    except ValueError as ve:
#        return -99 # np.zeros(len(vel[0]))


# In[98]:


obs_vel = [[3,3,3],[4,4,4],[5,5,5]]
obs_az = [30, 40, 10]
grid_az = [5, 10, 15, 20, 25, 30, 35, 40]
#myzz = ttfunc(obs_az, obs_vel, grid_az)
#display(myzz)
#myzz = ttfunc_partial(myx)

# vel data are sparse
# range is a vector to put on meshgrid, or to tile (range_dim_segment)
# az is a vector to put on a meshgrid, or to tile (az_dim_segment)
# the problem becomes, how to put vel data, which are sparse onto the az,range mesh?
# send an index along with the vel data; index is the azimuth vector
# vel and azimuth are both indexed by time.


# In[99]:


import xarray as xr
import xradar as xd

# real data
dirname = "/Users/brenda/data/PRECIP"   # MICA_testing"
# SEA20220702_001310_ppi.nc	SEA20220702_003301_ppi.nc	SEA20220702_004910_ppi.nc

filename = "SEA20220702_001310_ppi.nc"
localfilename = dirname + "/" + filename
#radar = xd.io.open_cfradial1_datatree(localfilename) # , engine="netcdf4" ) #first_dim="auto")
#radar = xd.io.open(localfilename)
ds = xr.open_dataset(localfilename) # , group="sweep_0") # , engine="nexradlevel2")
display(ds)


# In[100]:


12*360


# In[101]:


ds_time = ds.time.data
print('sweeps: ', ds.sweep_number.data, 'len=', len(ds.sweep_number.data))
display(ds.sweep_start_ray_index.data)
display(ds.sweep_start_ray_index[3].data)
display(ds.VEL2[0:10].data, len(ds.VEL2[0:10].data))   # VEL2 (time,range)
print(ds.azimuth.data, len(ds.azimuth.data))


# In[102]:


ds_range = ds.range.data
#display(ds_range.size)
ds_az = ds.azimuth.data
ds_el = ds.sweep_number.data
#display(ds_az)
az_dim_segment = np.tile(ds_az, ds_range.size)
range_dim_segment = np.tile(ds_range, ds_az.size)
#vel = ds.VEL.data
#vel = np.nan_to_num(vel, nan=-np.inf)
#velf = vel.flatten(order='F')
#display(velf)
vel_index = ds_range.size * ds_az.size


# In[107]:


azimuths = ds_az # [0, 90, 180, 270]

fields = ["DBZ2", "VEL2"]

elevations = ds_el[0:2] # [0.5, .75, 1.0]

dateTimes = [0] # TODO: this should be the number of files


# Elevation vs. Field grid
# APAR : panel (4) vs field grid
gridspace = hv.GridSpace(kdims=['Field', 'Elevation (Sweep)'], group='Parameters', label='Moments')
#gridspace = hv.GridSpace(kdims=['Amplitude', 'Power'], group='Parameters', label='Sines')

for elevation in elevations:
    sweep_start_ray_index = ds.sweep_start_ray_index[elevation].data
    sweep_end_ray_index = ds.sweep_end_ray_index[elevation].data
    nelements = sweep_end_ray_index - sweep_start_ray_index + 1
    ds_az = ds.azimuth[sweep_start_ray_index:nelements].data
    for field in fields:
        if (field == "VEL2"):
            obs_field = ds.VEL2[sweep_start_ray_index:nelements].data
        else:
            obs_field = ds.DBZ2[sweep_start_ray_index:nelements].data
        obs_field = np.nan_to_num(obs_field, nan=-np.inf)

        #holomap = hv.HoloMap(kdims='Frequency')
        holomap = hv.HoloMap(kdims='DateTime')
        for dateTime in dateTimes:
            # open and read data file
            # load az, field data ... TODO: it would be better to rearrange the looping because
            # the elevation and fields are in the same data file, but is this possible?
            # ---
            az = np.linspace(0, 360, 361)
            #r = np.linspace(150.0, 7.5e+04, round((7.5e+04-150)/150.0) + 1) # 150.0 300.0 ... 7.485e+04 7.5e+04
            r = ds_range # np.linspace(1, 3, 3)
            #print(r)
            xx, yy = np.meshgrid(r, az) # , sparse=True)
            #print('el=', elevation, ' ', field, 'dateTime=', dateTime, ' : ', xx.shape, yy.shape)
            
            #grid_az = az
            #obs_vel = vel
            obs_az = ds_az
            zz = ttfunc(obs_az, obs_field, az, len(r))
            #print('zz.shape=', zz.shape)
            #print("yy.shape=", yy.shape)
            #print("xx.shape=", xx.shape)
            #print("zz.shape=", zz.shape)
            
            azflat = np.hstack(yy.ravel())
            rangeflat = np.hstack(xx.ravel())
            zzflat = np.hstack(zz.ravel())
            #print(azflat.shape, rangeflat.shape, zzflat.shape)
            radar_df = pd.DataFrame({"values":zzflat, "az":azflat, "range":rangeflat})
            # ---
            #sines = {phase : hv.Curve(sine_curve(phase, field, amplitude, power))
            #         for phase in phases}
            # ppi = {azimuth : hv.Curve(sine_curve(azimuth, field, amplitude, power))
            #         for azimuth in azimuths}
            # have ppi_plot(return a dataframe)
            # radar_df = ttfunc(obs_az, obs_vel, grid_az)
            # --- ---
            ppiHeatmap = hv.HeatMap(radar_df, ["az", "range"])
            ppiHeatmap.opts(opts.HeatMap(cmap='jet', radial=True, xmarks=4, ymarks=6))
            #
            #ndoverlay = hv.NdOverlay(ppiHeatmap, kdims='Elevation').relabel(group='Elevations',
            #                                                       label='PPI', depth=1)
            #overlay = ndoverlay #  * hv.Points([(i,0) for i in range(0,10)], group='Markers', label='Dots')
            #holomap[frequency] = overlay 
            holomap[dateTime] = ppiHeatmap # overlay # associate date-time index with this grid of elev vs field
        gridspace[field, elevation] = holomap

#penguins = hv.RGB.load_image('../reference/elements/assets/penguins.png').relabel(group="Family", label="Penguin")

layout = gridspace # + penguins.opts(axiswise=True)


# In[108]:


layout


# In[111]:


# read, plot, write NEXRAD online data (without downloading the file)?
import xarray as xr
import xradar as xd
import cmweather
import matplotlib.pyplot as plt
import pyproj
#import cartopy
import hvplot.xarray


# examples from Ryan May and openradar (siphon and THREDDS Data Server)
# Ryan May: https://nbviewer.org/gist/dopplershift/356f2e14832e9b676207
# openradar: 

# Yah, there is the Amazom S3 and then Google Cloud.
# Google Cloud has a REST API for partial data reads.  
# Google Cloud does not have as much data as Amazon S3.
# To access Google Cloud, I need to login to my Google account
# https://console.cloud.google.com/storage/browser/gcp-public-data-nexrad-l2

# https://storage.googleapis.com/storage/v1/b/my-bucketT/o/my-object?fields=id,name,metadata/key1

    


# In[112]:


# dirname = "/Users/brenda/data/from_Alex/for_hawkeye_demo/Goshen_tornado_2009/DOW6/20090605";
dirname = "/Users/brenda/data/APAR/from_brad/polar"   # MICA_testing"
# need to send the earth radius because the latitudes don't vary
# enough for the default calculation
#
#filename = "cfrad.20090605_223048.231_to_20090605_223055.341_DOW6_PPI.nc"

#dirname = "/Users/brenda/data"
#filename = "cfrad.20161006_190650.891_to_20161006_191339.679_KAMX_SUR.nc"
# need to untar and gunzip .Z files
filename = "cfrad.20010101_010000.006_to_20010101_010000.744_APAR_sim_AIR.nc" # "KTLX19910606_075656"
localfilename = dirname + "/" + filename
#radar = xd.io.open_cfradial1_datatree(localfilename) # , engine="netcdf4" ) #first_dim="auto")
#radar = xd.io.open(localfilename)
ds = xr.open_dataset(localfilename) # , group="sweep_0") # , engine="nexradlevel2")
display(ds)

# TODO make up some data: Top, Bottom, Left, Right, etc.
# polar vs. cartesian, 
# grid layout
# Top (aft beam to 30 vertical, vertical to down 35 degrees)
# Left Right (Aft beam to 30; 35 to aft beam)
# Bottom (nadir to 35, horizontal to 70 degrees down)
#ds = xr.open_dataset(filename, group="sweep_0", engine="cfradial1")
#display(ds)

#dtree = xd.io.open_cfradial1_datatree(
#    localfilename)
#,
#    first_dim="time",
#    optional=False,
#)
#display(dtree)
#dtree.

#type(dtree["sweep_0"]) # .ds.DBZ.sortby("azimuth").plot(y="azimuth")


# In[113]:


dspolar = ds

import numpy as np
import holoviews as hv
from holoviews import dim

hv.extension('matplotlib')


# In[114]:


dspolar1 = dspolar.sel(time='2001-01-01T01:00:00.006000041')
display(dspolar)

# time(68)
# range(500)
# azimuth(time)
# VEL (time, range)

display(dspolar.azimuth.data)
display(dspolar.time.data)

#theta = dspolar.
#scatter = hv.Scatter((theta, r), 'theta', 'r').redim(r=dict(range=(0,2.5)))


# In[95]:


t = 30
az = dspolar.azimuth[t]
display("az=", az.data)

vel_at_time_t = dspolar.VEL[t]
display(type(vel_at_time_t.data))
vel2 = np.nan_to_num(vel_at_time_t, nan=-99)
display(vel2)


# In[115]:


# try another example from here https://holoviews.org/reference/elements/bokeh/RadialHeatMap.html
#
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[56]:


opts.defaults(opts.HeatMap(radial=True, width=800, height=800, tools=["hover"]))


# In[197]:


display(ds_az)
min(abs(np.diff(ds_az))) # az are NOT evenly spaced !


# In[230]:


b = np.zeros((3,4))
b[2] = [3,3,3,3]
b


# In[244]:


# hour ==> azimuth; day ==> range
ds_range = ds.range.data
#display(ds_range.size)
ds_az = ds.azimuth.data
#display(ds_az)
az_dim_segment = np.tile(ds_az, ds_range.size)
range_dim_segment = np.tile(ds_range, ds_az.size)
vel = ds.VEL.data
vel = np.nan_to_num(vel, nan=-np.inf)
velf = vel.flatten(order='F')
#display(velf)
vel_index = ds_range.size * ds_az.size

#radar_df = pd.DataFrame({"values": velf, "az": az_dim_segment, "range": range_dim_segment })

# a quick sanity test ...
# az(4) range(2) values(8)
#radar_df = pd.DataFrame({"values": [0,1,2,-99,4,5,6,7], "az": [0,90,180,358,0,90,180,358], "range": [5,5,5,5,10,10,10,10]})
# radar_df = pd.DataFrame({"values": [0,1,2,-99,4,5,6,8], "az": [0,10,20,30,0,10,20,30], "range": [5,5,5,5,10,10,10,10]})
# Try using a meshgrid of the precision and the extent we want
# and then embed the actual data into the meshgrid.
az = np.linspace(0, 360, 361)
#r = np.linspace(0, 5, 6) # , sparse=True)
r = np.linspace(150.0, 7.5e+04, round((7.5e+04-150)/150.0) + 1) # 150.0 300.0 ... 7.485e+04 7.5e+04
xx, yy = np.meshgrid(r, az) # , sparse=True)

grid_az = az
obs_vel = vel
obs_az = ds_az
zz = ttfunc(obs_az, obs_vel, grid_az)
print("yy.shape=", yy.shape)
print("xx.shape=", xx.shape)
print("zz.shape=", zz.shape)

azflat = np.hstack(yy.ravel())
rangeflat = np.hstack(xx.ravel())
zzflat = np.hstack(zz.ravel())
radar_df = pd.DataFrame({"values":zzflat, "az":azflat, "range":rangeflat})
#
#


# In[245]:


heatmap = hv.HeatMap(radar_df, ["az", "range"])
heatmap.opts(opts.HeatMap(cmap='jet', radial=True, xmarks=4, ymarks=6))


# In[ ]:


# ------------


# In[50]:


# use opendap; generate a url to fetch some data ...
# taken from here ... https://help.hydroshare.org/apps/thredds-opendap/
#ds = xr.open_dataset("http://thredds.hydroshare.org/thredds/dodsC/hydroshare/resources/f3f947be65ca4b258e88b600141b85f3/data/contents/SWE_time.nc?time[0:1:2183],y[0:1:58],x[0:1:38],transverse_mercator,SWE[0:1:0][0:1:58][0:1:38]")
ds = xr.open_dataset("http://thredds.hydroshare.org/thredds/dodsC/hydroshare/resources/f3f947be65ca4b258e88b600141b85f3/data/contents/SWE_time.nc?time[0:1:2183],y[0:1:58],x[0:1:38],SWE[0:1:0][0:1:58][0:1:38]")
# ds.to_netcdf("swe-t0.nc")
display(ds)
# YES!!! This is a partial query of data!!!


# In[51]:


display(ds.coords["x"])
display(ds.coords["y"])


# In[32]:


#ds.SWE[0][0][0]
#ds.time[0]
#print(type(ds))

#ds1d = ds.sel(x=433970.9, y=4616916.5) # time='2008-10-01T00:00:00.000000000')
#print(type(ds1d))
ds.hvplot()


# In[54]:


ds2 = ds.sel(time='2008-10-01T00:00:00.000000000')
print(type(ds2))

# convert NaN to something ...
ds3 = ds2.fillna(0)
#display(ds3.SWE.data)
ds3.hvplot('x','y', tiles=True)


# In[10]:


radar = radar.xradar.georeference()
display(radar)
#display(radar.children)

#radar = xd.georeference.get_x_y_z_tree(radar)
#display(radar["sweep_0"])


# In[ ]:




