import pandas as pd
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import pystan
from ggplot import *
import math
import re
import json
from datetime import datetime
from scipy import interpolate

pd.options.display.max_rows = 2000
pd.options.display.max_columns = 100

#default plot functions:
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["axes.labelsize"]= 20
#plt.rcParams['figure.savefig.dpi'] = 100
plt.rcParams['savefig.edgecolor'] = "#f2f2f2"
plt.rcParams['savefig.facecolor'] ="#f2f2f2"
plt.rcParams["figure.figsize"] = [15,8]
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 20
greens = ['#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
multi =['#66c2a4','#1f78b4','#a6cee3','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f']
plt.rcParams["axes.color_cycle"] = multi
#functions:

#show categories in a json file
def show_categories(json_data):
	dimensions = json_data["dataset"]["dimension"]
	time = dimensions.pop("Tid", None)
	return(dimensions)

#convert to datetime
def convert_datetime(date_series):
	if "K" in date_series[0]:
		time = date_series.apply(lambda x: x.replace("K",""))
		month = [str(int(t[-1])*3) for t in time]
		year = [str(int(t[:-1])) for t in time]
		new_date_series = [x+y for x,y in zip(year, month)]
		new_date_series = pd.to_datetime(new_date_series, format='%Y%m')
	if "M" in date_series[0]:
		time = date_series.apply(lambda x: x.replace("M",""))
		new_date_series = pd.to_datetime(time, format='%Y%m')
		return(new_date_series)

def format_df(df):
	"""
	columns should be labeled with "time" for the date and 
	"value" for the date. 
	inserts na values and converts value to float type. 

	"""
	df["time"] = convert_datetime(df.time)
	df["value"][df.value == ".."] =np.nan
	df["value"] = df.value.astype(float)
	return(df)



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




#Housing: long time series from norgesbank
#1912 = 100
xls_housing = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/house_price_index/p1c9.xlsx")
house_prices_l = xls_housing.parse('Table_A1', header=2)
house_prices_l.columns = ["year", "total", "oslo", "bergen", "trondheim", "kristiansand"]
house_prices_l = house_prices_l.iloc[:-15,:]

house_prices_melt = pd.melt(house_prices_l, id_vars="year")
house_prices_melt["value"][house_prices_melt.value==" "] = np.nan
house_prices_melt["value"] =house_prices_melt.value.astype(float)

fig, ax = plt.subplots()
hp_by_city = house_prices_melt.groupby("variable")
for city_price in hp_by_city:
	ax.plot(city_price[1].year, city_price[1].value, label=city_price[0])
	ax.text(2016, np.array(city_price[1].value)[-2], city_price[0])
ax.legend()
ax.set_ylabel("Houseprice index, 1912=100, log scale")
plt.show()



#credit
debt_json = pd.read_json("https://data.ssb.no/api/v0/dataset/62264.json?lang=no")
debt_cat = show_categories(debt_json)
debt_cat

debt = pd.read_csv("https://data.ssb.no/api/v0/dataset/62264.csv?lang=no", sep=";")
debt.columns = ['currency', 'sector', 'credit_source', 'time', 'variable',
       'value']

debt["time"] = convert_datetime(debt.time)
debt["value"][debt.value == ".."] = np.nan
debt["value"] = debt.value.astype(float)
debt["value"] = debt["value"]/1000

by_source = debt.groupby("credit_source")
for source in by_source:
	print(source[0])

by_sector = debt.groupby("sector")
for sect in by_sector:
	print(sect[0])

total_sources = debt[debt.credit_source=="LTOT Kredittkilder i alt"]
total_sources = total_sources[total_sources.currency=="00 I alt"]
total_sources["value"] = total_sources.value.astype(float)

fig, ax = plt.subplots()
tot_by_sector = total_sources.groupby("sector")
for sect in tot_by_sector:
	ax.plot(sect[1].time, sect[1].value, label=sect[0])
#ax.legend()
ax.annotate("Total", xy=(yearmonth("200801"),3500), size=14)
ax.annotate("Households", xy=(yearmonth("200801"), 2200), size=14)
ax.annotate("Non-financial firms", xy=(yearmonth("200801"), 1300), size=14)
ax.annotate("Municipalities", xy=(yearmonth("200801"), 500), size=14)
ax.set_ylabel("Gross debt, billions NOK")
fig.set_size_inches(15,8)
#fig.savefig("figures/debt_by_sector.png")
plt.show()


#household sector and non-financial firms by source

debt = debt[debt.value.notnull()]
debt = debt[debt.value!=0]

household = debt[debt.sector=="Kred04 Husholdninger mv."]
household = household[household.currency =="00 I alt"]

firms = debt[debt.sector=="Kred03 Ikke-finansielle foretak"]
firms = firms[firms.currency =="00 I alt"]

source_inc = ["L201 Statlige l�neinstitutter",
"L202 Banker",
"L203 Kredittforetak",
"LTOT Kredittkilder i alt"]

household = household[household.credit_source.isin(source_inc)]

fig, ax = plt.subplots()
household_by_source = household.groupby("credit_source")
for source in household_by_source:
	ax.plot(source[1].time, source[1].value, label=source[0])
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("Household gross debt, billions NOK")
fig.set_size_inches(15,8)
plt.show()

#household debt by currency
household = debt[debt.sector=="Kred04 Husholdninger mv."]
household = household[household.credit_source == "LTOT Kredittkilder i alt"]

fig, ax = plt.subplots()
household_by_currency = household.groupby("currency")
for currency in household_by_currency:
	ax.plot(currency[1].time, currency[1].value, label=currency[0])
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("Household gross debt, billions NOK")
fig.set_size_inches(15,8)
plt.show()

#non-financial debt by source:

firms_inc = ["L201 Statlige l�neinstitutter",
"L202 Banker",
"L203 Kredittforetak",
"L204 Finansieringsselskaper",
"L206 Livsforsikringsselskaper",
"L207 Skadeforsikringsselskaper",
"L209 Pensjonskasser",
"L210 Obligasjonsgjeld",
"L211 Sertifikatgjeld",
"L212 Andre kilder",
"LTOT Kredittkilder i alt"]

#firms = firms[firms.credit_source.isin(source_inc)]

fig, ax = plt.subplots()
firms_by_source = firms.groupby("credit_source")
for source in firms_by_source:
	print(source[0])
	ax.plot(source[1].time, source[1].value, label=source[0])
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("firms gross debt, billions NOK")
fig.set_size_inches(15,8)
plt.show()


#Interest rates
ir=pd.read_csv("http://www.norges-bank.no/WebDAV/stat/en/renter/v2/renter_mnd.csv")


#Trade and exchange

#exchange rate
pd.ExcelFile("http://www.norges-bank.no/en/Statistics/Historical-monetary-statistics/Historical-exchange-rates/"

xls_exchange = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/historical_exchange_rates/p1_c7.xlsx")
exchange_rates = xls_exchange.parse('p1_c7_Table_A2', header=2)


#oil and gas

#prices from eia
xls = pd.ExcelFile("http://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls")
brent_prices = xls.parse('Data 1', header=2)

brent_prices.columns = ["date", "brent_price"]
brent_prices["date"] = pd.to_datetime(brent_prices.date, format="%Y-%m-%d")

fig, ax = plt.subplots()
ax.plot(brent_prices.date, brent_prices.brent_price, label="Brent Crude Price, $/Barrel")
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("Brent oil price, USD / Barrel")
plt.show()


#prices from NPD
tot_prod=pd.read_csv("http://factpages.npd.no/ReportServer?/FactPages/TableView/field_production_totalt_NCS_month__DisplayAllRows&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=CSV&Top100=false&IpAddress=158.37.94.56&CultureCode=nb-no")
tot_prod.columns = ['﻿prod_year', 'prod_month', 
'oil_millm3', 'gas_billsm3',
'ngl_millsm3', 'condensate_millsm3',
'oe_millsm3', 'water_millsm3']

tot_prod["date"] = pd.to_datetime(tot_prod.loc[:, '﻿prod_year'].astype(str) + tot_prod.loc[:,"prod_month"].astype(str),
	format = "%Y%m")

prod_include = ['date', 'oil_millm3', 'gas_billsm3','water_millsm3']
tot_prod = tot_prod[prod_include]

tot_prod_long = pd.melt(tot_prod, id_vars =["date"])
tot_prod_long["value"] = tot_prod_long.value.astype(float)
tot_prod_long = tot_prod_long[tot_prod_long.value!=0]

def smooth_series(srs):
	x_range = [i for i in range(len(srs))]
	smoothed = sm.nonparametric.lowess(srs,x_range, frac=0.1)
	return(pd.Series(smoothed[:,1]))

smoothed = tot_prod_long.groupby("variable")["value"].transform(smooth_series)
tot_prod_long["smoothed"] = smoothed

fig, ax = plt.subplots()
prod_by_liquid = tot_prod_long.groupby("variable")
for liquid in prod_by_liquid:
	ax.plot(liquid[1].date, liquid[1].value, label=liquid[0])
for liquid in prod_by_liquid:
	ax.plot(liquid[1].date, liquid[1].smoothed)
ax.legend()
#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("Production")
plt.show()

#in MillSm3

#Investments in oil/gas
investments=pd.read_csv("http://factpages.npd.no/ReportServer?/FactPages/TableView/field_investment_yearly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=CSV&Top100=false&IpAddress=158.37.94.112&CultureCode=en")

tot_investments = investments.groupby("prfYear")['prfInvestmentsMillNOK'].aggregate(sum)
tot_investments = tot_investments.reset_index()
tot_investments.columns = ["year", "invest_millNOK"]
tot_investments = tot_investments[tot_investments.invest_millNOK!=0]

fig, ax = plt.subplots()
ax.bar(tot_investments.year, tot_investments.invest_millNOK)
plt.show()


reserves=reserves = read.csv("http://factpages.npd.no/ReportServer?/FactPages/TableView/field_reserves&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=CSV&Top100=false&IpAddress=158.37.94.112&CultureCode=en", stringsAsFactors=FALSE)




