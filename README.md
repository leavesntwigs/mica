# mica
CIDD in python

hvplot - built on holoviews - interactive layer on top of other packages: matplotlib, bokeh, plotly
panel  - built on Bokeh

mamba install geoviews, hvplot, panel


channels:
    - conda-forge 
# and the packages to install
dependencies:
    - numpy
    - xarray
    - xradar
    - panel
    - holoviews
    - jupyter
----

using this environment ...

mamba activate bokeh_heatmap_env
jupyter lab

trying to fix the no image display issue in jupyter notebooks

mamba create
-n holoviews_panel_20241007
-c conda-forge
Numpy xarray xradar panel holoviews jupyter

but, it still does not display an image!

-----

to run panel as web app

 panel serve panel_file_chooser_cards.ipynb --autoreload

----

fetch data ...

  /scr/cirrus3/rsfdata/projects/precip

corresponding images are here
  $PROJ_DIR/images/spol_moments/qc1/v1.0/20220525

---

to convert notebook to py file....

jupyter nbconvert --to script panel_file_chooser_cards_datatree.ipynb 

open browser here 
Bokeh app running at: http://localhost:5006/panel_file_chooser_cards
