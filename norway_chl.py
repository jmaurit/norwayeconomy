#norway_chl.py
#class

import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.io.shapereader as shpreader

#class norway_chl(plt)???
class norway_chl(object):
	"""Takes a dataframe with a single series with floats per 
	municipality in norway with 4-digit identifier. Outputs
	Chloropleth map. 
	arguments:
	df = dataframe with dataset
	geo = name of column with SSB region identifier
	value = name of column with value series to be plotted
	ex: 
	norway_chl(df = mydf, geo = "region", value = "value")
	"""

	def __init__(self, df, geo = "region", value = "value"):
		self.df = df
		self.geo = geo
		self.value = value
		self.make_geo_dict()
		#self.read_shp()
		self.make_color_map()


	def make_geo_dict(self):
		"""
		takes dataframe and region
		"""
		kode = self.df[self.geo].str.split(" ").str.get(0)
		self.geo_dict = dict(zip(kode, self.df[self.value].values))
		return(self.geo_dict)

	def read_shp(self, adm = "munic"):
		"""Input 'munic', 'county', or 'country' for level of 
		administrative detail. default is 'munic'. returns iterative
		reader file"""
		if adm == "munic":
			shape_file = 'norway_shape/kartverket/kommuner/kommuner'
		elif adm == "county":
			shape_file = "norway_shape/NOR_adm_shp/NOR_adm1"
		elif adm == "country":
			shape_file = "norway_shape/NOR_adm_shp/NOR_adm0"
		else:
			shape_file = 'norway_shape/NOR_adm_shp/NOR_adm0'
		
		reader_file = shpreader.Reader(shape_file)
		return(reader_file)

	def make_color_map(self, num_colors=10, col_scheme = "Greens"):
		if col_scheme == "Greens":
			self.cmap = plt.cm.Greens
		elif col_scheme == "PRGn":
			self.cmap = plt.cm.PRGn
		elif col_scheme == "YlGn":
			self.cmap = plt.cm.YlGn
		elif col_scheme == "ocean":
			self.cmap = plt.cm.ocean
		else:
			self.cmap =plt.cm.Greens

		vmin = self.df[self.value].min()
		vmax = self.df[self.value].max()
		self.norm = mpl.colors.Normalize(vmin = vmin, vmax=vmax)
		self.bins = np.linspace(vmin, vmax, num_colors)
		return([self.cmap, self.norm, self.bins])

	def draw_map(self, title="title", legend_precision = 0):
		"""
		draws maps
		"""
		subplot_kw = dict(projection=ccrs.Mercator())

		self.figure = plt.subplots(figsize=(10, 14),
		                       subplot_kw=subplot_kw)
		self.figure[1].set_frame_on(False)
		self.figure[1].outline_patch.set_visible(False)
		self.figure[1].background_patch.set_alpha(0)
		#figure[0] = fig
		#figure[1] = ax
		self.figure[1].set_extent((2, 32.0, 56, 71))

		#counties
		reader_file = self.read_shp("munic")
		for princ, rec in zip(reader_file.geometries(), reader_file.records()):
		    kode = rec.attributes["komm"]
		    if kode<1000:
		        kode = "0" + str(kode)
		    else:
		        kode = str(kode)
		    color= self.cmap(self.norm(self.geo_dict[kode]))      
		    self.figure[1].add_geometries(princ, ccrs.PlateCarree(), facecolor=color, edgecolor="none")

		#Norway- boundary
		reader_norway = self.read_shp("country")
		norway = reader_norway.geometries()
		norway_geom = next(norway)
		self.figure[1].add_geometries(norway_geom, ccrs.PlateCarree(), facecolor="none", edgecolor='black', alpha=.5)
		
		#cities:
		cities = ["Bergen", "Oslo", "Trondheim", "Stavanger", "Kristiansand", "Tromsø"]
#(long, lat)
		coords = [[5.3221, 60.3913], [10.7522, 59.9139], [10.3951, 63.4305], [5.7331, 58.9700], 
		[8.0182, 58.1599], [18.9553, 69.6492]]

		city_coords = zip(cities, coords)
		for city, coords in city_coords: 
			plt.text(coords[0],coords[1], city, horizontalalignment='right', transform=ccrs.Geodetic())
		#legend
		#[*left*, *bottom*, *width*,*height*]
		#where
		ax_legend = self.figure[0].add_axes([0.7, 0.3, 0.03, 0.3], zorder=3)
		ticks = np.round(self.bins[1:-1], legend_precision)
		cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=self.cmap, norm=self.norm, ticks=ticks, boundaries=self.bins, orientation='vertical')
		cb.ax.set_title(title)

		return(self.figure)
	
	def save_plot(self, file_path):
		self.figure[0].savefig(file_path, bbox_inches='tight')




