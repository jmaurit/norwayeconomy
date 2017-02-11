#http://ramiro.org/notebook/basemap-choropleth/

#cartopy library for mapping
#From python for research
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader

#example using cartopy:

#innvandrer data
innvand = pd.read_csv("http://data.ssb.no/api/v0/dataset/48644.csv?lang=en", sep=",", na_values = [".", ".."])
befolk = pd.read_csv("http://data.ssb.no/api/v0/dataset/1108.csv?lang=en", sep=",", na_values = [".", ".."])

innvand16 = innvand[innvand.time == 2016]

tot_innvand16 = innvand16.groupby("region")["Immigrants by region, sex, country background and time"].aggregate(sum)

befolk16 = befolk[befolk.contents == "Population at the beginning of the quarter"]
befolk16 = befolk16.iloc[:,[0,3]]
befolk16 = befolk16.set_index("region")

immig = pd.concat([befolk16, tot_innvand16], axis=1)
immig.columns = ["population", "immigrants"]
immig.loc[:,"population"] = immig.population.astype(float)
immig["immig_perc"] = immig.immigrants/immig.population*100

birddata = pd.read_csv("bird_tracking.csv")
bird_names = np.unique(birddata.bird_name)

proj = ccrs.Mercator()

plt.figure(figsize = (10,10))
ax = plt.axes(projection=proj)
ax.set_extent([0, 30.0, 55, 70])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
plt.show()


#now with shapefile
#from
# https://ocefpaf.github.io/python4oceanographers/blog/2015/02/02/cartopy_folium_shapefile/
kw = dict(resolution='10m', category='cultural',
          name='admin_1_states_provinces')

states_shp = shpreader.natural_earth(**kw)

shp = shpreader.Reader(states_shp)


subplot_kw = dict(projection=ccrs.Mercator())

fig, ax = plt.subplots(figsize=(12, 15),
                       subplot_kw=subplot_kw)
ax.set_extent((0, 30.0, 52, 72))

for record, state in zip(shp.records(), shp.geometries()):
    if record.attributes['admin'] == 'Norway':
          ax.add_geometries([state], ccrs.PlateCarree(), facecolor="gray", edgecolor='black')
plt.show()


#read files from folder and start to plot
#read shapefile
shape_country = 'norway_shape/NOR_adm_shp/NOR_adm0'
shape_county = 'norway_shape/NOR_adm_shp/NOR_adm1'
shape_princ = 'norway_shape/kartverket/kommuner/kommuner'

reader_prins = shpreader.Reader(shape_princ)
norway_prins= reader_prins.geometries()
#match with data
income = pd.read_csv("http://data.ssb.no/api/v0/dataset/49678.csv", sep=";")

income_2015 = income[income.tid==2015]
income_2015 = income_2015[income_2015.husholdningstype =="0000 Alle husholdninger"]
income_2015 = income_2015[income_2015.statistikkvariabel =="Inntekt etter skatt, median (kr)"]
income_2015 = income_2015.iloc[:,[0,4]]
income_2015.columns = ["region", "income"]

kode = income_2015.region.str.split(" ").str.get(0)
income_2015_dict = dict(zip(kode, income_2015.income.values))

#next(norway_counties)

reader_country = shpreader.Reader(shape_country)
#countries = reader_country.records()
#country = next(countries)
norway = reader_country.geometries()
norway_geom = next(norway)

norway_rec = reader_country.records()
norway_record = next(norway_rec)


subplot_kw = dict(projection=ccrs.Mercator())
fig, ax = plt.subplots(figsize=(15, 20),
                       subplot_kw=subplot_kw)
ax.set_extent((0, 30.0, 52, 72))
#ax.add_geometries(norway_geom, ccrs.Mercator(), facecolor="grey", edgecolor='black')
for p in princ:
  ax.add_geometries(p, ccrs.PlateCarree(), facecolor="grey", edgecolor='black')
plt.show()

for country in countries:
  print(country.attributes["ADMIN"])

population = lambda country: country.attributes['pop_est']

# sort the countries by population and get the first 5
countries_by_pop = sorted(reader.records(), key=population)[:5]

geometries = reader_country.geometries()

reader = shpreader.Reader('ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces')
counties = reader.records()

for county in counties:
  #print(county.attributes["adm0_name"])
  print(county.attributes["name"])

county = next(counties)
print(type(county.attributes))
print(sorted(county.attributes.keys()))

ax.background_patch.set_visible(False)
ax.outline_patch.set_visible(False)

plt.title('counties')

for county in shpreader.Reader(counties_shp).geometries():
    # pick a default color for the land with a black outline,
    # this will change if the storm intersects with our track
    facecolor = [0.9375, 0.9375, 0.859375]
    edgecolor = 'black'
    facecolor = '#FF7E00'
    ax.add_geometries([county], ccrs.PlateCarree(),
                  facecolor=facecolor, edgecolor=edgecolor)
plt.show()




import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyjstat import pyjstat #for reading json-stat format
import requests
from collections import OrderedDict

from geonamescache import GeonamesCache
#from helpers import slug
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

pd.options.display.max_rows = 999
pd.options.display.max_columns = 50


gc = GeonamesCache()
iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())

df = pd.read_csv(filename, skiprows=4, usecols=cols)
df.set_index('Country Code', inplace=True)
df = df.ix[iso3_codes].dropna() # Filter out non-countries and missing values.

values = df[year]
cm = plt.get_cmap('Greens')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
df['bin'] = np.digitize(values, bins) - 1
df.sort('bin', ascending=False).head(10)

#mpl.style.use('map')
fig = plt.figure(figsize=(22, 12))

#ax = fig.add_subplot(111, axisbg='w', frame_on=False)
ax = fig.add_subplot(111)
fig.suptitle('Forest area as percentage of land area in {}'.format(year), fontsize=30, y=.95)

m = Basemap(lon_0=0, projection='robin')
m.drawmapboundary(color='w')

m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
for info, shape in zip(m.units_info, m.units):
  iso3 = info['ADM0_A3']
  if iso3 not in df.index:
      color = '#dddddd'
  else:
      color = scheme[df.ix[iso3]['bin']]

  patches = [Polygon(np.array(shape), True)]
  pc = PatchCollection(patches)
  pc.set_facecolor(color)
  ax.add_collection(pc)

# Cover up Antarctica so legend can be placed over it.
ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)

# Draw color legend.
#left, bottom, width, height
ax_legend = fig.add_axes([0.90, 0.10, 0.03, 0.4], zorder=3)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='vertical')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

# Set the map footer.
plt.annotate(descripton, xy=(-1.85, -3.2), size=14, xycoords='axes fraction')

plt.show()


#maps.py
#global administrative areas
#http://www.gadm.org/download
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

#from BeautifulSoup import BeautifulSoup

#import data on housing prices by county
house_prices = pd.read_csv("https://data.ssb.no/api/v0/dataset/25138.csv?lang=no", sep=";")
house_prices.columns = ["region", "type", "year", "variable", "price_sq_m"]

house_prices = house_prices[house_prices.type=="02 Brukte eneboliger"]
house_prices = house_prices[house_prices.variable == "Kvadratmeterpris (kr)"]
house_prices = house_prices[house_prices.year==2014]

fylke = house_prices.region.values

fylke2 = ["Østfold", "Akershus", "Oslo", "Hedmark",
 "Oppland", "Buskerud", "Vestvold", "Telemark", "Aust-Agder",
  "Vest-Agder", "Rogaland", "Hordaland", "Sogn og Fjordane", 
  "Møre og Romsdal", "Sør-Trøndelag", "Nord-Trøndelag", "Nordland", 
  "Troms", "Finnmark"] 
fylke_dict = dict(zip(fylke, fylke2))

house_prices["counties"] = [fylke_dict[county] for county in house_prices.region]
del house_prices["region"]
["price_sq_m", "counties"]


fig=plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111)

#map instance
m = Basemap(llcrnrlon=3.,llcrnrlat=57.,\
      urcrnrlon=33.,urcrnrlat=72.,\
            resolution='i',projection='merc')
            #lat_0=2.5,lon_0=58.)

m.readshapefile('norway_shape/NOR_adm_shp/NOR_adm0', "norway")
m.readshapefile('norway_shape/NOR_adm_shp/NOR_adm1', "counties")

shapes=[]
names = []
for info, shape in zip(m.counties_info, m.counties):
  shapes.append(shape)
  names.append(info["NAME_1"])

names_series = pd.Series(names)
names_series = names_series[~names_series.duplicated()]

names_series2 = ["Østfold"]
for f in names_series[1:]:
  names_series2.append(f)

shape_fylke_dict2 = dict(zip(names_series, names_series2))

#now create dataframe with shapefiles and names
df_shape = pd.DataFrame()
df_shape["counties"] = [shape_fylke_dict2[name] for name in names]
df_shape["shapes"] = shapes

df_shape = df_shape.merge(house_prices[["price_sq_m", "counties"]], on="counties")
df_shape["price_sq_m"] = df_shape["price_sq_m"].astype(float)

#create different colors
num_colors = 9
values = df_shape["price_sq_m"]
cm = plt.get_cmap('Greens')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
df_shape['bin'] = np.digitize(values, bins) - 1
#df.sort('bin', ascending=False).head(10)

def add_polys(shape_row):
  color = scheme[shape_row.bin]
  patches = [Polygon(np.array(shape_row.shapes), True)]
  pc = PatchCollection(patches)
  pc.set_facecolor(color)
  ax.add_collection(pc) 

df_shape.apply(add_polys, axis=1)

plt.show()


#add municipalities:
#innflytting siste 9 kvartaler
#new_res = pd.read_csv("https://data.ssb.no/api/v0/dataset/1106.csv?lang=eng", sep=";")



res_url = "https://data.ssb.no/api/v0/dataset/1106.json?lang=no"
data = requests.get(res_url)
res_data = pyjstat.from_json_stat(data.json(object_pairs_hook=OrderedDict))

new_res = res_data[0]
new_res.columns = ["princ", "variable", "quarter", "value"]
folkevekst = new_res[new_res.variable=="Folkevekst"]
folkevekst15 = folkevekst[folkevekst.quarter == "2015K3"]
folketall = new_res[new_res.variable =="Folketalet ved utgangen av kvartalet"]
folketall15 = folketall[folketall.quarter =="2015K3"]
folkevekst15 = folkevekst15.merge(folketall15[["princ", "value"]], on="princ")
folkevekst15.columns = ["princ", "variable", "quarter", "folkevekst", "folketall"]
folkevekst15["folkevekst_perc"] = 100* folkevekst15.folkevekst/folkevekst15.folketall

#get kommune num to match with shape data
ssb_kommune = pd.read_csv("ssb_kommune.csv")
del ssb_kommune["Unnamed: 0"]
ssb_kommune.columns =["kode", "princ"]

folkevekst15 = folkevekst15.merge(ssb_kommune, on="princ", how="left")




fig=plt.figure(figsize=(15,10))
ax  = fig.add_subplot(111)

#map instance
m = Basemap(llcrnrlon=3.,llcrnrlat=57.,\
      urcrnrlon=33.,urcrnrlat=72.,\
            resolution='i',projection='merc')
            #lat_0=2.5,lon_0=58.)

m.readshapefile('norway_shape/NOR_adm_shp/NOR_adm0', "norway")
m.readshapefile('norway_shape/kartverket/kommuner/kommuner', "princ", drawbounds=False)

shapes=[]
kode = []
names = []
for info, shape in zip(m.princ_info, m.princ):
  shapes.append(shape)
  kode.append(info["komm"])
  names.append(info["navn"])

df_folkevekst15 = pd.DataFrame()
df_folkevekst15["shapes"] = shapes
df_folkevekst15["kode"] = kode

#merge with ssb data
df_folkevekst15 = df_folkevekst15.merge(folkevekst15, on="kode", how="left")

#create different colors for map
num_colors = 20
values = df_folkevekst15["folkevekst_perc"].astype(float)
cm = plt.get_cmap('Greens')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
df_folkevekst15['bin'] = np.digitize(values, bins) - 1
df_folkevekst15.sort_values('bin', ascending=False).head(10)

def add_polys(shape_row):
  color = scheme[shape_row.bin]
  patches = [Polygon(np.array(shape_row.shapes), True)]
  pc = PatchCollection(patches)
  pc.set_facecolor(color)
  ax.add_collection(pc) 

df_folkevekst15.apply(add_polys, axis=1)


#princ_dict = dict(zip(shape_princ.values, data_princ.values))

plt.show()



#Data on county moving

pop_dyr =  pd.read_csv("http://data.ssb.no/api/v0/dataset/49577.csv?lang=no", sep=";")

folkemengde = pop_dyr[pop_dyr.statistikkvariabel=="Folkemengde"]
folkemengde.columns = ["region", "date", "variable", "population"]
       'Folkemengde 1. januar og endringer i kalender�ret, etter region, tid og statistikkvariabel'
folkemengde["population"] = folkemengde.population.astype(float)

def first_diff(block, periods):
  diff_block = block.diff(periods)
  return(diff_block)


pop_diff=folkemengde.groupby("region")["population"].transform(first_diff, 1)
pop_diff5 = folkemengde.groupby("region")["population"].transform(first_diff, 5)
folkemengde["pop_diff"] = pop_diff
folkemengde["pop_diff5"] = pop_diff5 
folkemengde = folkemengde[folkemengde.pop_diff.notnull()]
folkemengde["perc_diff"] = folkemengde.pop_diff/folkemengde.population*100
folkemengde["perc_diff5"] = folkemengde.pop_diff5/folkemengde.population*100

