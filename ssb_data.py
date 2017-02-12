import pandas as pd
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import pystan
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


#valg deltakelse

valg2015 = pd.read_csv("https://www.ssb.no/eksport/tabell.csv?key=271517", sep=";", decimal=",", na_values = [".", ".."])
del valg2015["Unnamed: 14"]


def biggestparty(komm):
	#print(komm)
	#komm = valg2015.iloc[0,:]
	return(np.argmax(komm[1:]))

#biggestparty(valg2015.iloc[0,:])

valg2015["biggest"] = valg2015.apply(biggestparty, axis=1)



#internasjonal handel

ih = pd.read_csv("http://data.ssb.no/api/v0/dataset/58962.csv?lang=no", 
	sep=";", decimal=",", na_values = [".", ".."])

ih.columns = ["flow", "date", "variable", "value"]
ih.date = convert_datetime(ih.date)
ih.flow.unique()

include = ['Etotusorn Fastlandseksport', 'Eoljegass Eksport av r�olje, naturgass og kondensater']
eksport  = ih[ih.flow.isin(include)]
eksport["value"] = eksport.value.astype(float)
eksport_w = eksport.pivot(index='date', columns='flow', values='value')
eksport_w.reset_index(inplace=True)

eksport_w.columns = ["date", "petroleum_export", "non_petroleum_export"]

eksport_w["perc_petroleum_export"] = eksport_w.petroleum_export.diff(periods=12)
eksport_w["smooth_petroleum_export"] = pd.rolling_mean(eksport_w.petroleum_export, window=12)
eksport_w["perc_non_petroleum_export"] = eksport_w.non_petroleum_export.diff(periods=12)
eksport_w["smooth_non_petroleum_export"] = pd.rolling_mean(eksport_w.non_petroleum_export, window=12)


fig, ax = plt.subplots()
ax.plot(eksport_w.date, eksport_w.petroleum_export)
ax.plot(eksport_w.date, eksport_w.non_petroleum_export)
plt.show()


include = ["Hbtotuso Handelsbalansen (Eksport - import, begge uten skip og oljeplattformer)", "Hbtusorn Handelsbalansen (Fastlandseksport - import utenom skip og oljeplattformer)"]
bal  = ih[ih.flow.isin(include)]
bal["value"] = bal.value.astype(float)
bal_w = bal.pivot(index='date', columns='flow', values='value')
bal_w.reset_index(inplace=True)

bal_w.columns = ["date", "trade_bal", "trade_bal_ex_pet"]
fig, ax = plt.subplots()
ax.plot(bal_w.date, bal_w.trade_bal)
ax.plot(bal_w.date, bal_w.trade_bal_ex_pet)
plt.show()

#Nasjonalregnskap

NR=pd.read_csv("http://data.ssb.no/api/v0/dataset/59022.csv", 
         sep=";", decimal=",", na_values = [".", ".."])
NR.columns = ["variabel", "kvartal", "enhet", "verdi"]

NR["kvartal"]  = convert_datetime(NR.kvartal) #fra zoo

#sesong justert
NRSA = NR[NR.enhet == "Faste 2013-priser, sesongjustert (mill. kr)"]

#enhet "Faste 2013-priser, sesongjustert (mill. kr)"
NRSA = NRSA[["variabel", "kvartal", "verdi"]]

#use tidyr to split
kateg = NRSA.variabel.str.split('.', n=1).str[0]
variabel = NRSA.variabel.str.split(' ', n=1).str[1]

NRSA["kateg"] = kateg
NRSA["variabel"] = variabel

#investering

investering = NRSA[NRSA.kateg =="bif"]
del investering["kateg"]
invest_w = investering.pivot(index='kvartal', columns='variabel', values='verdi')

invest_var = ['Total Investment', 'Fixed Assets',
       'Mainland Norway',
       'Foreign Shipping',
       'Extraction',
       'Mainland Norway, excl public sector',
       'Housing',
       'Private Sector',
       'Public Sector',
       'Other services',
       'Other production',
       'Industry and Mining',
       'Extraction Services']

invest_w.columns = invest_var

invest_stack = invest_w[["Extraction", "Housing", "Public Sector", 'Mainland Norway', "Total Investment"]]
invest_stack["Other"] = invest_stack["Total Investment"] - invest_stack[["Extraction", "Housing", "Public Sector", 'Mainland Norway']].sum(axis=1)

del invest_stack["Total Investment"]

# invest_stack.reset_index(inplace=True)

# invest_stack = pd.melt(invest_stack, id_vars = "kvartal")

annotate = pd.dataframe()

Y = np.array(invest_stack)

dates = investering.kvartal.unique()

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)
plt.show()


#Se på eksport or import

eksport = NRSA[NRSA.kateg == "eks"]
del eksport["kateg"]
eksport_w = eksport.pivot(index='kvartal', columns='variabel', values='verdi')
eksport_w.columns =['Total', 'Oil and Gas',
       'Ships and Platforms', 'Services',
       'Traditional Commodities']
eksport_stack = eksport_w
del eksport_stack["Total"]


Y = np.array(eksport_stack)

dates = eksport_stack.index.values

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)
plt.show()

#imports

imports =  NRSA[NRSA.kateg == "imp"]

del imports["kateg"]
imports_w = imports.pivot(index='kvartal', columns='variabel', values='verdi')
imports_w.columns =['Total', 'Oil and Gas',
       'Ships and Platforms', 'Services',
       'Other Goods']

imports_stack = imports_w
del imports_stack["Total"]


Y = np.array(imports_stack)

dates = imports_stack.index.values

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)
plt.show()


#Privat og offentlig konsum

pkon = NRSA[NRSA.kateg=="koh"]

del pkon["kateg"]
pkon_w = pkon.pivot(index='kvartal', columns='variabel', values='verdi')
pkon_w.columns =['Household and Nonprofit',
       'Household', 'Household Foreign Consumption',
       'Services', 'Foreigners Consumption in Norway', 'Goods']

pkon_stack = pkon_w[['Services', 'Goods']]


Y = np.array(pkon_stack)

dates = pkon_stack.index.values

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)
plt.show()


okon = NRSA[NRSA.kateg=="koo"]

del okon["kateg"]
okon_w = okon.pivot(index='kvartal', columns='variabel', values='verdi')
okon_w.columns =['Public sector', 
	  'Principalities',
       'State',
       'State, Military',
       'State, Civil']

okon_stack = okon_w[['Principalities',
       'State, Military',
       'State, Civil']]

Y = np.array(okon_stack)

dates = okon_stack.index.values

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)
plt.show()

bnp = NRSA[NRSA.kateg == "bnpb"]
bnp = bnp[bnp.variabel=='Bruttonasjonalprodukt, markedsverdi']
bnp = bnp[["kvartal", "verdi"]]
bnp.columns = ["kvartal", "bnp"]

okon_stack.reset_index(inplace=True)
okon_perc = okon_stack.merge(bnp, how="left", on="kvartal")
okon_perc["State, Military, %GDP"] = okon_perc["State, Military"]/okon_perc["bnp"]*100
okon_perc["State, Civil, %GDP"] = okon_perc["State, Civil"]/okon_perc["bnp"]*100
okon_perc["Principalities, %GDP"] = okon_perc["Principalities"]/okon_perc["bnp"]*100
okon_perc = okon_perc[['kvartal', 'State, Military, %GDP', 'Principalities, %GDP',
       'State, Civil, %GDP']]

okon_l = pd.melt(okon_perc,id_vars="kvartal")

fig, ax = plt.subplots()
start = datetime.strptime('01012008', '%d%m%Y')
okon_by_sector = okon_l.groupby("variable")
for sector in okon_by_sector:
	ax.plot(sector[1].kvartal, sector[1].value, label=sector[0])
	#ax.annotate(unemp[0], xy=(start, np.array(sector[1].value)[0]+1))
ax.legend()
ax.set_ylabel("Size of Public Sector, % of GDP")
#fig.savefig("figures/unemployment.png")
plt.show()

laks_eksport = pd.read_csv("http://data.ssb.no/api/v0/dataset/1122.csv?lang=no", sep=";")

#Demographics - immigration:

pop = pd.read_csv("http://data.ssb.no/api/v0/dataset/49626.csv?lang=no", sep=";")

pop.columns = ["region", "year", "variable", "value"]
pop = pop[["year", "variable", "value"]]
pop_tot = pop[pop.variable=="Folkemengde"]
pop_tot = pop_tot[["year", "value"]]
pop_tot.columns = ["year", "pop_tot"]

innvandring = pd.read_csv("http://data.ssb.no/api/v0/dataset/48651.csv?lang=no", sep=";")
innvandring.columns = ['region', 'kjoenn', 'landbakgrunn', 'year', 'statistikkvariabel',
       'innvandrer']
innvandring = innvandring[['kjoenn', 'landbakgrunn', 'year',
       'innvandrer']]
innvandring = innvandring.merge(pop_tot, on="year", how ="left")
innvandring["pros_innvan"] = innvandring["innvandrer"]/innvandring["pop_tot"]*100

total = innvandring.groupby(["year", "landbakgrunn"])["pros_innvan"].aggregate(sum)
total = total.reset_index()
total_w = total.pivot(index="year", columns="landbakgrunn", values="pros_innvan")
Y = np.array(total_w)
x=total.year.unique()
fig, ax = plt.subplots()
ax.stackplot(x, Y.T)
plt.show()
#Demographics - population:

fm5aar=pd.read_csv("http://data.ssb.no/api/v0/dataset/65195.csv?lang=no", sep=";")
fm5aar.columns = ['alder', 'kjoenn', 'tid', 'statistikkvariabel',
       'personer']
fm5aar["alder"] = [i.split(" ")[1] for i in fm5aar.alder]


totalt = fm5aar[fm5aar.kjoenn == "0 Begge kj�nn"]

totalt_fm5 = totalt[["tid", "alder", "personer"]]
totalt_fm5_w = totalt_fm5.pivot(index='tid', columns='alder', values='personer')
totalt_fm5_w.columns.values
totalt_fm5_w = totalt_fm5_w[['0-4','5-9', '10-14','15-19', '20-24', '25-29', '30-34', '35-39',
       '40-44', '45-49',  '50-54', '55-59', '60-64', '65-69',
       '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100']]

Y = np.array(totalt_fm5_w)

dates = totalt_fm5.tid.unique()

fig, ax = plt.subplots()
ax.stackplot(dates, Y.T)

plt.show()



#employment
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


#Housing cost_index

cost_index = pd.read_csv("http://data.ssb.no/api/v0/dataset/26944.csv?lang=no", sep=";", decimal=",", na_values=["..", "."])
cost_index.columns = ['type', 'date', 'variable','index']
cost_index["variable"].levels
cost_index["date"] = pd.to_datetime(cost_index.date, format="%YM%m")

tot_cost_index = cost_index[cost_index.type=="01 I alt"]
del tot_cost_index["type"]

tot_cost_index_w = tot_cost_index.pivot(index="date", columns = "variable", values="index")
tot_cost_index_w.columns = ['cost index', '% change, mom',
       '% change, yoy']
tot_cost_index_w.reset_index(inplace=True)

fig, ax = plt.subplots(2)
ax[0].plot(tot_cost_index_w.date, tot_cost_index_w["cost index"])
ax[1].plot(tot_cost_index_w.date, tot_cost_index_w["% change, yoy"])
plt.show()

okon.pivot(index='kvartal', columns='variabel', values='verdi')

#house prices
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

#compare prices with building costs
tot_price_index = house_prices[house_prices.type == "00 Boliger i alt"]
tot_price_index = tot_price_index[tot_price_index.region =="Total"]
tot_price_index = tot_price_index.pivot(index="time", columns="region", values="value")
del tot_price_index["type"]
del tot_price_index["variable"]
del tot_price_index["region"]
tot_price_index["% change, yoy"] = tot_price_index.value.pct_change(periods=4)*100

fig, ax = plt.subplots()
ax.plot(tot_cost_index_w.date, tot_cost_index_w["% change, yoy"])
ax.plot(tot_price_index.time, tot_price_index["% change, yoy"])
ax.set_xlim([pd.to_datetime("1993-01-01"), pd.to_datetime("2017-01-01")])
plt.show()


#Looking at movement to the cities

#Oslo population, by part of city
oslo_pop = pd.read_csv("http://data.ssb.no/api/v0/dataset/1010.csv?lang=no",sep=";",  decimal=",")
oslo_pop["grunnkrets"] = oslo_pop.grunnkrets.astype("category")

#Looking at building in the cities
bygging = pd.read_csv("http://data.ssb.no/api/v0/dataset/26025.csv?lang=no", sep=";", header=0, decimal=",")
bygging["region"]=bygging.region.astype('category')



bygging["region"] = bygging.region.cat.rename_categories(fylker)

bygging["statistikkvariabel"] = bygging.statistikkvariabel.astype("category")
bygging.statistikkvariabel.cat.categories
variabler = ['Boliger under arbeid', 'Fullfoert bruksareal (1 000 m2)',
       'Fullfoerte boliger', 'Igangsatt bruksareal (1 000 m2)',
       'Igangsatte boliger', 'Under arbeid, bruksareal (1 000 m�)']
bygging["statistikkvariabel"] = bygging.statistikkvariabel.cat.rename_categories(variabler)
bygging.rename(columns={"Boligbygg, etter region, statistikkvariabel og tid": "boligbygging"}, inplace=True)
bygging["tid"] = pd.to_datetime(bygging.tid, format="%YM%m")
bygging["boligbygging"]=bygging.boligbygging.astype(float)


igangsatte = bygging[bygging.statistikkvariabel == "Igangsatte boliger"]
rel_fylker = ['Oslo', 'Rogaland', 'Hordaland',  'Soer-Troendelag']
igangsatte = igangsatte[bygging.region.isin(rel_fylker)]
igangsatte["region"] = igangsatte.region.astype('str')

from scipy.interpolate import UnivariateSpline

igangsatte = igangsatte[igangsatte.boligbygging.notnull()]

#bygging_by_city = igangsatte.groupby("region")
Oslo = igangsatte[igangsatte.region=="Oslo"]
Hordaland = igangsatte[igangsatte.region=="Hordaland"]
Rogaland = igangsatte[igangsatte.region =="Rogaland"]
Soer_Troendelag = igangsatte[igangsatte.region=="Soer-Troendelag"]

#smooth the city data
d = {"date":igangsatte["tid"][igangsatte.region=="Oslo"]}
smooth_data = pd.DataFrame(data=d)
t = len(city[1].tid)
T = [i for i in range(t)]

s_city = UnivariateSpline(T, Oslo.boligbygging, k=3, s=6000000)
smooth_data["Oslo"] = s_city(T)

s_city = UnivariateSpline(T, Hordaland.boligbygging, k=3, s=2000000)
smooth_data["Hordaland"] = s_city(T)

s_city = UnivariateSpline(T, Rogaland.boligbygging, k=3, s=2000000)
smooth_data["Rogaland"] = s_city(T)

s_city = UnivariateSpline(T, Soer_Troendelag.boligbygging, k=3, s=2000000)
smooth_data["Soer-Troendelag"] = s_city(T)

smooth_data_l = pd.melt(smooth_data, id_vars = "date")

smooth_by_city = smooth_data_l.groupby("variable")

bygging_by_city = igangsatte.groupby("region")

fig, ax = plt.subplots(4)
for a, city in enumerate(smooth_by_city):
	ax[a].plot(city[1].date, city[1].value)
	ax[a].set_ylabel(city[0], size=14)

for a, city in enumerate(bygging_by_city):
	ax[a].plot(city[1].tid, city[1].boligbygging, alpha=.2, color="blue")
	#n = len(region[1].value)
#ax.set_ylabel("Number of new dwellings", size=14)
fig.set_size_inches(6,8)
fig.set_label("New housing starts, by principality with major city")
#fig.savefig("figures/city_new_dwellings.png")
plt.show()


fylker = ['Oestfold', 'Akershus', 'Oslo', 'Hedmark', 'Oppland',
       'Buskerud', 'Vestfold', 'Telemark', 'Aust-Agder',
       'Vest-Agder', 'Rogaland', 'Hordaland',
       'Bergen', 'Sogn og Fjordane', 'Moere og Romsdal',
       'Sør-Trøndelag', 'Nord-Troendelag', 'Nordland',
       'Troms', 'Finnmark']


#Look at population
population = pd.read_csv("http://data.ssb.no/api/v0/dataset/49623.csv?lang=no", sep=";")
population["tid"] = pd.to_datetime(population.tid, format="%Y")

population.region = population.region.astype('category')

population["region"] = population.region.cat.rename_categories(fylker)
population = population[population.region.isin(['Oslo', 'Rogaland', 'Hordaland','Sør-Trøndelag'])]

population.columns = ["region", "tid", "statistikkvariabel", "population"]
population["population"] = population.population.astype(float)
population.region = population.region.astype(str)

innflyttinger = population[population.statistikkvariabel.isin(["Innflyttinger"])]
del innflyttinger["statistikkvariabel"]
innflyttinger.columns = ['region', 'tid', 'innflyttinger']
innflyttinger["utflyttinger"] = population.population[population.statistikkvariabel=="Utflyttinger"].values
innflyttinger["net_migration"] = innflyttinger["innflyttinger"] - innflyttinger["utflyttinger"]
folkemengde = population[population.statistikkvariabel=="Folkemengde"]

innflyttinger_by_city = innflyttinger.groupby("region")
fig, ax = plt.subplots()
for city in innflyttinger_by_city:
	ax.plot(city[1].tid.iloc[10:-1], city[1].net_migration.iloc[10:-1], label=city[0])
ax.text(pd.to_datetime(2002, format="%Y"), 6000, "Oslo")
ax.text(pd.to_datetime(1975, format="%Y"), 2500, "Rogaland")
ax.text(pd.to_datetime(2006, format="%Y"), 800, "Sør-Trøndelag")
ax.text(pd.to_datetime(1965, format="%Y"), 3000, "Hordaland")
ax.set_ylabel("Net migration, principalities with major cities")
plt.show()

folkemengde_by_city = folkemengde.groupby("region")
for city in folkemengde_by_city:
	ax[1].plot(city[1].tid.iloc[20:-1], city[1].population.iloc[20:-1], label=city[0])
#ax[1].text(pd.to_datetime(1997, format="%Y"), 34000, "Oslo")
#ax[1].text(pd.to_datetime(1965, format="%Y"), 3000, "Rogaland (Stavanger)")
#ax[1].text(pd.to_datetime(1997, format="%Y"), 4000, "Sør-Trøndelag (Trondheim)")
#ax[1].text(pd.to_datetime(1997, format="%Y"), 14000, "Hordaland (Bergen)")
ax[1].set_ylabel("Population")

plt.show()

#Hele landet

bygging = pd.read_csv("http://data.ssb.no/api/v0/dataset/95146.csv?lang=no", sep=";", header=0, decimal=",")
bygging["statistikkvariabel"] = bygging.statistikkvariabel.astype("category")
bygging.statistikkvariabel.cat.categories
variabler = ['Boliger under arbeid', 'Fullfoert bruksareal (1 000 m2)',
       'Fullfoerte boliger', 'Igangsatt bruksareal (1 000 m2)',
       'Igangsatte boliger', 'Under arbeid, bruksareal (1 000 m�)']
bygging["statistikkvariabel"] = bygging.statistikkvariabel.cat.rename_categories(variabler)
bygging.rename(columns={"Boligbygg, etter region, statistikkvariabel og tid": "boligbygging"}, inplace=True)
bygging["tid"] = pd.to_datetime(bygging.tid, format="%YM%m")
bygging["boligbygging"]=bygging.boligbygging.astype(float)

igangsatte = bygging[bygging.statistikkvariabel == "Igangsatte boliger"]
t = len(igangsatte.tid)
T = [i for i in range(t)]

s_igangsatte = UnivariateSpline(T, igangsatte.boligbygging, k=3, s=50000000)
smooth_igangsatte = s_igangsatte(T)


fig, ax = plt.subplots()
ax.plot(igangsatte.tid, igangsatte.boligbygging, alpha=.5)
ax.plot(igangsatte.tid, smooth_igangsatte)
ax.set_ylabel("New housing starts")
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
	ax.plot(source[1].time, source[1].value, label=source[0])
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("firms gross debt, billions NOK")
fig.set_size_inches(15,8)
plt.show()






#building costs
apartment_build_cost = pd.read_csv("https://data.ssb.no/api/v0/dataset/1058.csv?lang=no", sep=";", decimal=",")
building_costs = pd.read_csv("https://data.ssb.no/api/v0/dataset/26944.csv?lang=no", sep=";", decimal=",")


apartment_build_cost = apartment_build_cost[apartment_build_cost.arbeidstype=="20 Bustadblokk i alt"]
yoy_apartbc =  apartment_build_cost[apartment_build_cost.statistikkvariabel=="Endring fr� f�rre �r (prosent)"]


apart_bc = apartment_build_cost[apartment_build_cost.statistikkvariabel=="Byggjekostnadsindeks"]

del apart_bc["arbeidstype"]
del apart_bc["statistikkvariabel"]

del yoy_apartbc["arbeidstype"]
del yoy_apartbc["statistikkvariabel"]

apart_bc["tid"] = pd.to_datetime(apart_bc.tid, format = "%YM%m")

apart_bc.columns = ["date", "ci"]
apart_bc["ci"]=apart_bc.ci.apply(lambda x: x.replace(",","."))
apart_bc["ci"] = apart_bc.ci.astype(float)

fig, ax = plt.subplots()
ax.plot(apart_bc.date, apart_bc.ci)
ax.legend()
	#ax.annotate(sect[0], xy=(start, np.array(sect[1].value[0])+5 ))
ax.set_ylabel("Building Cost Index")
fig.set_size_inches(15,8)
plt.show()


#Interest rates
#Norwegian Overnight Weighted Average rate.

ir=pd.read_csv("http://www.norges-bank.no/WebDAV/stat/en/renter/v2/renter_mnd.csv")

ir["DATES"] = [datetime.strptime(d, "%b-%y") for d in ir.DATES]

ir.columns = ['date', 'folio_nom', 'res_nom', 'dlaan_nom',
	   'statskvl_3m_eff','statskvl_6m_eff', 'statskvl_9m_eff', 
	   'statskvl_12m_eff','statsobl_3y_eff', 'statsobl_5y_eff', 
	   'statsobl_10y_eff', 'nowa_rt','nowa_vl']



include = ['date','dlaan_nom', 'statskvl_3m_eff', 'statskvl_12m_eff', 'statsobl_10y_eff']
ir = ir[include]

ir_long = pd.melt(ir, id_vars="date")
ir_long["value"][ir_long.value=="ND"] = np.nan
ir_long["value"] = ir_long.value.astype(float)

fig, ax = plt.subplots()
for r in ir_long.groupby("variable"):
	ax.plot(r[1].date, r[1].value, label=r[0])
ax.legend()
plt.show()



nibor = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/short_term_interest_rates/NIBOR_dag_mnd_aar.xlsx")
long_ir = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/short_term_interest_rates/p2_c1-c7.xlsx")
real_ir = long_ir.parse('p2c7_table_7B1',header=2)
real_ir = real_ir.iloc[:-13,:]
real_ir.Year = real_ir.Year.astype(float)

rir_include = ['Year', 'Real marginal rate', 'Real deposit rate', 'Real loans rate',
       'Real bond yield']

inflation_include = ['Year', 'Inflation rate', 'Smoothed inflation rate']

inflation = real_ir[inflation_include]
real_ir = real_ir[rir_include]

real_ir_long = pd.melt(real_ir, id_vars="Year")
fig, ax = plt.subplots()
for r in real_ir_long.groupby("variable"):
	ax.plot(r[1].Year, r[1].value, label=r[0])
ax.legend()
plt.show()

inflation_long = pd.melt(inflation, id_vars="Year")
fig, ax = plt.subplots()
for i in inflation_long.groupby("variable"):
	ax.plot(i[1].Year, i[1].value, label=i[0])
ax.legend()
plt.show()
#Trade and exchange

#exchange rate
xls_exchange = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/historical_exchange_rates/p1_c7.xlsx")
exchange_rates = xls_exchange.parse('p1_c7_Table_A2', header=2)



#wages
wages = pd.ExcelFile("http://www.norges-bank.no/Upload/HMS/wages_by_industry/p2c6_7.xlsx")
tot_wages = wages.parse('Table_total', header=2)
ind_wages = wages.parse('Table_6A4', header=3)

#total wages
tot_wages = tot_wages.iloc[:-12,:]
tot_wages["Year"][tot_wages.Year=="2014*"] = "2014"
tot_wages["Year"] = tot_wages.Year.astype(int)

tot_wages_long = pd.melt(tot_wages, id_vars="Year")

fig, ax = plt.subplots()
for wage in tot_wages_long.groupby("variable"):
	ax.plot(wage[1].Year, wage[1].value, label=wage[0])
ax.legend()
plt.show()
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


#Import/Export
#Export of goods:
exports_by_type = pd.read_csv("https://data.ssb.no/api/v0/dataset/34256.csv?lang=en")





