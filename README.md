# mica
CIDD in python

mamba install geoviews, hvplot, panel

----

using this environment ...

mamba activate bokeh_heatmap_env
jupyter lab

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
