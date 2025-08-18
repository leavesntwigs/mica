import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd


# read cartesian radar data
filename = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521/ncf_20220521_151748.nc"



#filename = "hail/2024080600_00_ODIMH5_PVOL6S_VOL_CASSM.h5"
filename = "hail/2024080602_30_ODIMH5_PVOL6S_VOL_CASSM.h5"
filename = "/Users/brenda/data/ams2025/titan/netcdf/Strathmore/titan_20240806.nc"
filename = "/Users/brenda/data/ams2025/titan/netcdf/KingCity/titan_20220521.nc"

# may need to read the data file using xarray?   can xradar read it?

# read only DBZH field, one sweep,  for all data files
#
ds = xr.open_dataset(filename, group="sweep_0", engine="odim")
print(ds)
# works to here


# trying the netcdf4-python package
>>> import netCDF4
>>> import numpy as np
>>> f = netCDF4.Dataset(filename)
>>> print(f)
This works.
<class 'netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    version: 1.0
    convention: TitanStormTracking
    title: 
    institution: 
    source: 
    comment: 
    dimensions(sizes): 
    variables(dimensions): int64 file_time(), int64 start_time(), int64 end_time(), int32 n_scans(), int32 sum_storms(), int32 sum_layers(), int32 sum_hist(), int64 sum_runs(), int64 sum_proj_runs(), int32 max_simple_track_num(), int32 max_complex_track_num()
    groups: scans, storms, tracks, gprops, layers, hist, runs, proj_runs, simple, complex, entries

To access a group:
>>> print(f.groups['storms'])
>>> print(f.groups['storms'].variables.keys())
dict_keys(['low_dbz_threshold',
 'high_dbz_threshold',
 'dbz_hist_interval',
 'hail_dbz_threshold',
 'base_threshold',
 'top_threshold',
 'min_storm_size',
 'max_storm_size',
 'morphology_erosion_threshold',
 'morphology_refl_divisor',
 'min_radar_tops',
 'tops_edge_margin',
 'z_p_coeff',
 'z_p_exponent',
 'z_m_coeff',
 'z_m_exponent',
 'sectrip_vert_aspect',
 'sectrip_horiz_aspect',
 'sectrip_orientation_error',
 'poly_start_az',
 'poly_delta_az',
 'check_morphology',
 'check_tops',
 'vel_available',
 'n_poly_sides',
 'ltg_count_time',
 'ltg_count_margin_km',
 'hail_z_m_coeff',
 'hail_z_m_exponent',
 'hail_mass_dbz_threshold',
 'gprops_union_type',
 'tops_dbz_threshold',
 'precip_computation_mode',
 'precip_plane_ht',
 'low_convectivity_threshold',
 'high_convectivity_threshold'])


# to access a variable: where f is a netCDF4.Dataset
>>> f.variables['file_time']




>>> print(f_King['complex'])
<class 'netCDF4.Group'>
group /complex:
    dimensions(sizes): max_complex(199)
    variables(dimensions): 
 int32 complex_track_nums(max_complex),
 int32 complex_track_num(max_complex),
 float32 volume_at_start_of_sampling(max_complex),
 float32 volume_at_end_of_sampling(max_complex),
 int32 start_scan(max_complex),
 int32 end_scan(max_complex),
 int32 duration_in_scans(max_complex),
 int32 duration_in_secs(max_complex),
 int64 start_time(max_complex),
 int64 end_time(max_complex),
 int32 n_simple_tracks(max_complex),
 int32 n_top_missing(max_complex),
 int32 n_range_limited(max_complex),
 int32 start_missing(max_complex),
 int32 end_missing(max_complex),
 int32 n_samples_for_forecast_stats(max_complex),
 float32 forecast_bias_proj_area_centroid_x(max_complex),
 float32 forecast_bias_proj_area_centroid_y(max_complex),
 float32 forecast_bias_vol_centroid_z(max_complex),
 float32 forecast_bias_refl_centroid_z(max_complex),
 float32 forecast_bias_top(max_complex),
 float32 forecast_bias_dbz_max(max_complex),
 float32 forecast_bias_volume(max_complex),
 float32 forecast_bias_precip_flux(max_complex),
 float32 forecast_bias_mass(max_complex),
 float32 forecast_bias_proj_area(max_complex),
 float32 forecast_bias_smoothed_proj_area_centroid_x(max_complex),
 float32 forecast_bias_smoothed_proj_area_centroid_y(max_complex),
 float32 forecast_bias_smoothed_speed(max_complex),
 float32 forecast_bias_smoothed_direction(max_complex),
 float32 forecast_rmse_proj_area_centroid_x(max_complex),
 float32 forecast_rmse_proj_area_centroid_y(max_complex),
 float32 forecast_rmse_vol_centroid_z(max_complex),
 float32 forecast_rmse_refl_centroid_z(max_complex),
 float32 forecast_rmse_top(max_complex),
 float32 forecast_rmse_dbz_max(max_complex),
 float32 forecast_rmse_volume(max_complex),
 float32 forecast_rmse_precip_flux(max_complex),
 float32 forecast_rmse_mass(max_complex),
 float32 forecast_rmse_proj_area(max_complex),
 float32 forecast_rmse_smoothed_proj_area_centroid_x(max_complex),
 float32 forecast_rmse_smoothed_proj_area_centroid_y(max_complex),
 float32 forecast_rmse_smoothed_speed(max_complex),
 float32 forecast_rmse_smoothed_direction(max_complex),
 float32 ellipse_forecast_n_success(max_complex),
 float32 ellipse_forecast_n_failure(max_complex),
 float32 ellipse_forecast_n_false_alarm(max_complex),
 float32 polygon_forecast_n_success(max_complex),
 float32 polygon_forecast_n_failure(max_complex),
 float32 polygon_forecast_n_false_alarm(max_complex)



# get the complex_track_num
>>> complex_track_num = f_King['complex/complex_track_nums'][0]
# use it as an index into the other properties
>>> f_King['complex/start_time'][complex_track_num]
f_King['complex/n_simple_tracks'][complex_track_num]
# use it to get the associated simple tracks

def get_simple_tracks(complex_track_num):
    offset = f_King['simple/simples_per_complex_offsets'][complex_track_num]
    n_simples = f_King['simple/n_simples_per_complex'][complex_track_num]
    # now get the simple track numbers associated with this complex track
    simple_tracks = f_King['simple/simples_per_complex'][offset:n_simples]
    return simple_tracks

# more general form
def get_simple_tracks(ds, complex_track_num):
    offset = ds['simple/simples_per_complex_offsets'][complex_track_num]
    n_simples = ds['simple/n_simples_per_complex'][complex_track_num]
    # now get the simple track numbers associated with this complex track
    simple_tracks = ds['simple/simples_per_complex'][offset:n_simples]
    return simple_tracks

# TODO walk the simple group and get the parents and children; generate tree??

# read cartesian radar data

>>> filename = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521/ncf_20220521_173550.nc"
>>> ds = xr.open_dataset(filename)
>>> z=np.nan_to_num(ds.VEL.data[0,17], nan=-32)
>>> fig = go.Figure(data=go.Heatmap(z=z, type='heatmap', colorscale='Viridis'))
>>> fig.show()



# dtree1 = xd.io.open_cfradial1_dataset(filename)
ds = xr.open_dataset(filename)
dtree1 = dtree1.xradar.georeference()
print(dtree1["sweep_0"].ds)
try:
    dtree2 = dtree1.xradar.georeference()
except Exception:
    print("Georeferencing failed!")
print(dtree2["sweep_0"].ds)

dtree1 = dtree1.xradar.map_over_sweeps(get_geocoords)
ds = dtree1["sweep_0"].to_dataset()

#
# I think we can do a scatterplot with square markers with cartesian data
#
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ds["DBZH"].plot(x="x", y="y", vmin=-10, vmax=75, cmap="plasma", ax=ax[0])
ds["DBZH"].plot(x="lon", y="lat", vmin=-10, vmax=75, cmap="plasma", ax=ax[1])
plt.show()


