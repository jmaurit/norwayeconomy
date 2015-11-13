import pandas as pd
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import pystan
import scipy
from ggplot import *
import math
import re
import json
from datetime import datetime

pd.options.display.max_rows = 2000
pd.options.display.max_columns = 100

from pandas.io.json import json_normalize

employment_json = pd.read_json("https://data.ssb.no/api/v0/dataset/1054.json?lang=en")

age_labels = employment_json["dataset"]["dimension"]["Alder"]
sex_labels = employment_json["dataset"]["dimension"]["Kjonn"]
content_labels = employment_json["dataset"]["dimension"]["ContentsCode"]

employment = pd.read_csv("https://data.ssb.no/api/v0/dataset/1054.csv?lang=en")

employment.columns = ["sex", "age", "contents", "time", "value"]

# employment_all_25_74 = employment[employment.sex=="0 Both sexes"]
# employment_all_25_74 =employment_all_25_74[employment_all_25_74.age=="25-74 25-74 years"]
# employment_all_25_74 = employment_all_25_74[["contents", "time", "unemp"]]
# employment_all_25_74.pivot(index='time', columns='contents', values='value')

unemp_1000 = employment[employment.contents == 'Unemployment (LFS) (1 000 persons), seasonally adjusted']
#unemp_1000 = unemp_1000[unemp_1000.sex=="0 Both sexes"]
unemp_1000 = unemp_1000[unemp_1000.age=="25-74 25-74 years"]
unemp_1000 = unemp_1000[["time", "sex", "value"]]

unemp_1000["time"] = unemp_1000.time.apply(lambda x:  x.replace("M", ""))
unemp_1000["time"] = pd.to_datetime(unemp_1000["time"], format="%Y%m")

unemp_1000["value"][unemp_1000.value == ".."] = np.nan
unemp_1000["value"] = unemp_1000.value.astype(float)

unemp_1000.columns =["date", "sex", "value"]
unemp_1000["sex"] = unemp_1000.sex.apply(lambda x: x[2:])

#unemp_1000.set_index("date", inplace=True)
#unemp_1000.reset_index(inplace=True)

#start_date = datetime.strptime('01012006', '%d%m%Y')
#end_date = datetime.strptime('01012016', '%d%m%Y')

fig, ax = plt.subplots()
start = datetime.strptime('01012008', '%d%m%Y')
unemp_by_sex = unemp_1000.groupby("sex")
for unemp in unemp_by_sex:
	ax.plot(unemp[1].date, unemp[1].value, label=unemp[0])
	ax.annotate(unemp[0], xy=(start, np.array(unemp[1].value)[0]+1))
ax.legend()
ax.set_ylabel("Norwegian unemployment, 1000s")
fig.set_size_inches(10,7)
#fig.savefig("figures/unemployment.png")
plt.show()

#employed persons
emp_perc_sa = employment[employment.contents == "Employed persons in per cent of the population, seasonally adjusted"]

emp_perc_sa = emp_perc_sa[emp_perc_sa.age=="25-74 25-74 years"]
emp_perc_sa = emp_perc_sa[["time", "sex", "value"]]

emp_perc_sa["time"] = emp_perc_sa.time.apply(lambda x:  x.replace("M", ""))
emp_perc_sa["time"] = pd.to_datetime(emp_perc_sa["time"], format="%Y%m")

emp_perc_sa["value"][emp_perc_sa.value == ".."] =np.nan
emp_perc_sa["value"] = emp_perc_sa.value.astype(float)

emp_perc_sa.columns =["date", "sex", "value"]
emp_perc_sa["sex"] = emp_perc_sa.sex.apply(lambda x: x[2:])

emp_perc_sa.set_index("date", inplace=True)
emp_perc_sa.reset_index(inplace=True)

start = datetime.strptime('01012000', '%d%m%Y')

fig, ax = plt.subplots()
emp_by_sex = emp_perc_sa.groupby("sex")
for emp in emp_by_sex:
	ax.plot(emp[1].date, emp[1].value, label=emp[0])
	ax.annotate(emp[0], xy=(start, np.array(emp[1].value[0])+5 ))
ax.set_ylabel("Norwegian Employment, %")
fig.set_size_inches(11,7)
plt.show()



#Bankruptcies***************************

def show_categories(json_data):
	print(json_data["dataset"]["dimension"]["ContentsCode"])

show_categories(bank_json)

bank_json = pd.read_json("https://data.ssb.no/api/v0/dataset/95265.json?lang=en")
bank_json["dataset"]["dimension"]["ContentsCode"]

bankruptcies = pd.read_csv("https://data.ssb.no/api/v0/dataset/95265.csv?lang=en")

enter_bank = bankruptcies[bankruptcies.contents == 'Bankruptcies related to enterprises (excl. sole propriertorships)']
pers_bank = bankruptcies[bankruptcies.contents == 'Personal bankruptcies (incl. sole propriertorships)']

def format_df(df):
	df.columns = ["time", "contents", "value"]
	df = df[["time", "value"]]
	df["time"] = df.time.apply(lambda x:  x.replace("M", ""))
	df["time"] = pd.to_datetime(df["time"], format="%Y%m")
	df["value"][df.value == ".."] =np.nan
	df["value"] = df.value.astype(float)
	return(df)

enter_bank = format_df(enter_bank)
pers_bank = format_df(pers_bank)

start = datetime.strptime('01012000', '%d%m%Y')

from scipy.interpolate import UnivariateSpline

enter_bank = enter_bank[enter_bank.value.notnull()]
pers_bank = pers_bank[pers_bank.value.notnull()]

t = len(enter_bank.time)
T = [i for i in range(t)]
s_enter = UnivariateSpline(T, enter_bank.value, k=3, s=400000)
smooth_enter = s_enter(T)
s_pers = UnivariateSpline(T, pers_bank.value, k=3, s=200000)
smooth_person = s_pers(T)


fig, ax = plt.subplots()
ax.plot(enter_bank.time, enter_bank.value, color="green", alpha=.5)
ax.plot(enter_bank.time, smooth_enter, color="green")
ax.plot(pers_bank.time, pers_bank.value, color="navy", alpha=.5)
ax.plot(pers_bank.time, smooth_person, color="navy")
ax.annotate('Personal Bankruptcies', xy=(start, 350), size=14)
ax.annotate('Enterprise Bankruptcies', xy=(start, 40), size=14)
ax.set_ylabel("Bankruptcies in Norway, per month", size=14)
fig.set_size_inches(10,8)
fig.savefig("figures/bank_plot.png")
plt.show()


#Boligpriser

house_prices = pd.read_csv("https://data.ssb.no/api/v0/dataset/1060.csv?lang=no", sep=";", header=0)
house_prices.columns = ['region', 'type', 'time', 'variable','value']
time = house_prices.time.apply(lambda x: x.replace("K",""))
month = [str(int(t[-1])*3) for t in time]
year = [str(int(t[:-1])) for t in time]
house_prices["time"] = [x+y for x,y in zip(year, month)]
house_prices["time"] = pd.to_datetime(house_prices.time, format='%Y%m')
na_values = ["..", "."]
house_prices = house_prices[~house_prices.value.isin(na_values)]
house_prices["value"] = house_prices.value.apply(lambda x: float(x.replace(",", ".")))

names = []
prices_by_region = house_prices.groupby("region")
for region in prices_by_region:
	names.append(region[0])

new_names = ["Oslo with Baerum", "Stavanger", "Bergen", 
"Trondheim", "Akershus", "Southeast", "Hedmark and Oppland",
"Agder and Rogaland", "Westcoast", "Troendelag", "Northern Norway",
"Total"]

names_dict = dict(zip(names,new_names))

house_prices["region"] = house_prices.region.apply(lambda x: names_dict[x])

cities = new_names[0:4]

house_prices_cities = house_prices[house_prices.region.isin(cities)]
total_cities = house_prices_cities[house_prices_cities.type == "00 Boliger i alt"]

#blah

start = datetime.strptime('01012005', '%d%m%Y')
end = datetime.strptime('01012015', '%d%m%Y')
total_cities = total_cities[total_cities.time>=start]

fig, ax = plt.subplots()
houses_by_city= total_cities.groupby("region")
for city in houses_by_city:
	ax.plot(city[1].time, city[1].value, label=city[0])
	ax.legend()
	#n = len(region[1].value)
ax.annotate("Trondheim", xy=(datetime.strptime('01012007', '%d%m%Y'),110)
ax.set_ylabel("Housing Prices, index, 2005 = 100", size=14)
fig.set_size_inches(11,7)
fig.savefig("figures/city_housing_prices.png")
plt.show()


#Police offences
police_offences = pd.read_json("https://data.ssb.no/api/v0/dataset/81192.json?lang=en")

