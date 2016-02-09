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



res_url = "https://data.ssb.no/api/v0/dataset/1106.json?lang=no"
data = requests.get(res_url)
res_data = pyjstat.from_json_stat(data.json(object_pairs_hook=OrderedDict))

new_res = res_data[0]
new_res.columns = ["princ", "variable", "quarter", "value"]
folkevekst = new_res[new_res.variable=="Folkevekst"]
folkevekst15 = folkevekst[folkevekst.quarter == "2015K3"]
data_princ = folkevekst.princ[~folkevekst.princ.duplicated()]

data_princ.sort_values(inplace=True)

kommune_num = pd.read_csv("http://hotell.difi.no/download/ssb/regioner/kommuner", sep=";")

kommune_num = kommune_num[["kode", "tittel"]]

folkevekst15 = kommune_num.merge(folkevekst15, left_on = "tittel", right_on = "princ", how="outer")

kommune_num_compare = folkevekst15[["kode", "tittel", "princ"]]
kommune_num_compare_comp = kommune_num_compare[kommune_num_compare.tittel.notnull()]
matched = kommune_num_compare_comp[kommune_num_compare_comp.princ.notnull()]

not_matched = kommune_num_compare[~kommune_num_compare["princ"].isin(matched["princ"])]
not_matched.to_csv("not_matched.csv", sep=";")

matched_rest = pd.read_csv("matched_rest.csv", sep=";")
del matched_rest["Unnamed: 0"]

ssb_kommune = matched.append(matched_rest)[["kode", "princ"]]
ssb_kommune.to_csv("ssb_kommune.csv")
