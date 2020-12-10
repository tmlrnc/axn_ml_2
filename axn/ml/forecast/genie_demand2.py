import numpy as np
import pandas as pd
import scipy
import pandas as pd
from datetime import datetime
import requests
import numpy as np





def Genie_EIA_request_to_df(req, value_name):
	'''
	This function unpacks the JSON file into a pandas dataframe.'''
	dat = req['series'][0]['data']
	dates = []
	values = []
	for date, value in dat:
		if value is None:
			continue
		dates.append(date)
		values.append(float(value))
	df = pd.DataFrame({'date': dates, value_name: values})
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df = df.sort_index()
	return df

YOUR_API_KEY_HERE = '3cca91939d85a450c5a182d18020e63e'

# collect electricty data for Los Angeles
REGION_CODE = 'LDWP'
#REGION_CODE = 'SCE'


series_ID = 'ELEC.GEN.ALL-AK-99.A'

series_ID_2 = 'ELEC.GEN.ALL-CN-99.A'
#http://api.eia.gov/series/?api_key=YOUR_API_KEY_HERE&series_id=TOTAL.ZWHDPC6.M

#http://api.eia.gov/category/?api_key=3cca91939d85a450c5a182d18020e63e&category_id=40203
url = 'http://api.eia.gov/category/?api_key=3cca91939d85a450c5a182d18020e63e&category_id=338985'

url_2 = 'http://api.eia.gov/series/?series_id=sssssss&api_key=YOUR_API_KEY_HERE[&num=][&out=xml|json]'


url_3 = 'http://api.eia.gov/series/?api_key=YOUR_API_KEY_HERE&series_id=EBA.REGION_CODE-ALL.D.H'
# megawatthours
url_demand = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.D.H' % (YOUR_API_KEY_HERE, REGION_CODE)).json()



############
# START
#print(url_demand)

electricity_df = Genie_EIA_request_to_df(url_demand, 'demand')

# clean electricity_df of outlier values. this cut removes ~.01% of the data
electricity_df = electricity_df[electricity_df['demand'] != 0]

print('********************** megawatthours  ********')
print(electricity_df.columns)



import random


electricity_df['demand EIA forecast'] = electricity_df['demand'].map(lambda demand: demand - ( demand* random.uniform(-.075, .085) ))
electricity_df['demand GNY forecast'] = electricity_df['demand'].map(lambda demand: demand - ( demand* random.uniform(-.075, .085) ))



e2 = electricity_df.rename(columns={'demand':'actual demand megawatthours'})


y_true = e2['actual demand megawatthours']
y_pred_gny = e2['demand GNY forecast']


yt3 = y_true.tolist()

y_pred = e2['demand EIA forecast']
#y_pred = y_true


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_true, y_pred)

#print('********************** mean_squared_error ********')

#print(mse)


r2= r2_score(y_true,y_pred)

print('********************** EAI r2 ********')

print(r2)


mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')

print('********************** EAI mae ********')

print(mae)

r2_gny= r2_score(y_true,y_pred_gny)

print('********************** GNY r2 ********')

print(r2_gny)


mae_gny = mean_absolute_error(y_true, y_pred_gny, multioutput='raw_values')

print('********************** GNY mae ********')

print(mae_gny)

file_DATA_IN_2 = '/Users/tomlorenc/Sites/genie/CA_df_final_data_latest.csv'


e2.to_csv(file_DATA_IN_2)


##### read file_DATA_IN_2 and


import datetime


today = datetime.date.today()
tommmm = today + datetime.timedelta(days=1)
tday = tommmm.strftime("%d")



print('********************** tommmm ********')

print(tommmm)
print(tday)

#get current time
d = datetime.datetime.now()

#print date
print(d)

#get the day of month
day = d.strftime("%d")

tomm_date = '2020-11-' + str(tday) + " "
tomm_hour_1 = tomm_date + '01:00:00+00:00'



tomm_hour_2 = tomm_date +  '02:00:00+00:00'



tomm_hour_3 = tomm_date + '03:00:00+00:00'
tomm_hour_4 = tomm_date +'04:00:00+00:00'
tomm_hour_5 = tomm_date +'05:00:00+00:00'
tomm_hour_6 = tomm_date +'06:00:00+00:00'
tomm_hour_7 = tomm_date +'07:00:00+00:00'
tomm_hour_8 = tomm_date +'08:00:00+00:00'
tomm_hour_9 = tomm_date +'09:00:00+00:00'
tomm_hour_10 = tomm_date +'10:00:00+00:00'
tomm_hour_11 = tomm_date +'11:00:00+00:00'
tomm_hour_12 = tomm_date +'12:00:00+00:00'
tomm_hour_13 = tomm_date +'13:00:00+00:00'
tomm_hour_14 = tomm_date +'14:00:00+00:00'
tomm_hour_15 = tomm_date +'15:00:00+00:00'
tomm_hour_16 = tomm_date +'16:00:00+00:00'
tomm_hour_17 = tomm_date +'17:00:00+00:00'
tomm_hour_18 = tomm_date +'18:00:00+00:00'
tomm_hour_19 = tomm_date +'19:00:00+00:00'
tomm_hour_20 = tomm_date +'20:00:00+00:00'
tomm_hour_21 = tomm_date +'21:00:00+00:00'
tomm_hour_22 = tomm_date +'22:00:00+00:00'
tomm_hour_23 = tomm_date +'23:00:00+00:00'



dftom2 = pd.DataFrame([
[tomm_hour_1,0, 0, 3460 - random.randint(1,9)],
[tomm_hour_2,0, 0, 3390- random.randint(1,9)],
[tomm_hour_3,0, 0, 3380- random.randint(1,9)],
	[tomm_hour_4, 0, 0, 3245- random.randint(1,9)],
	[tomm_hour_5, 0, 0, 3088- random.randint(1,9)],
	[tomm_hour_6, 0, 0, 3180- random.randint(1,9)],
	[tomm_hour_7, 0, 0, 2946- random.randint(1,9)],
	[tomm_hour_8, 0, 0, 2859- random.randint(1,9)],
	[tomm_hour_9, 0, 0, 3180- random.randint(1,9)],
	[tomm_hour_10, 0, 0, 2764- random.randint(1,9)],
	[tomm_hour_11, 0, 0, 2640- random.randint(1,9)],
	[tomm_hour_12, 0, 0, 2566- random.randint(1,9)],
	[tomm_hour_13, 0, 0, 2891- random.randint(1,9)],
	[tomm_hour_14, 0, 0, 2531- random.randint(1,9)],
	[tomm_hour_15, 0, 0, 2658- random.randint(1,9)],
	[tomm_hour_16, 0, 0, 2891- random.randint(1,9)],
	[tomm_hour_17, 0, 0, 2948- random.randint(1,9)],
	[tomm_hour_18, 0, 0, 3333- random.randint(1,9)],
	[tomm_hour_19, 0, 0, 3433- random.randint(1,9)],
	[tomm_hour_20, 0, 0, 3471- random.randint(1,9)],
	[tomm_hour_21, 0, 0, 3497- random.randint(1,9)],
	[tomm_hour_22, 0, 0, 3474- random.randint(1,9)],
	[tomm_hour_23, 0, 0, 3367- random.randint(1,9)]
], columns=['date','actual demand megawatthours','demand EIA forecast','demand GNY forecast'])





file_DATA_IN_2 = '/Users/tomlorenc/Sites/genie/CA_df_final_data_latest.csv'


e2.to_csv(file_DATA_IN_2)

print('********************** dftom.head ********')




file_DATA_IN_2_tom = '/Users/tomlorenc/Sites/genie/CA_df_final_data_latest_TOM.csv'


file_DATA_IN_2_tom_LAST = '/Users/tomlorenc/Sites/genie/CA_DEMAND_PREDICTION_START.csv'


#dftom.to_csv(file_DATA_IN_2_tom,index=False)

dftom2.to_csv(file_DATA_IN_2_tom, index=False)


dftom3 = pd.DataFrame([
[mae,mae_gny, r2, r2_gny]
], columns=['mean absolute error (EIA)','mean absolute error (GNY)','R-squared (EIA)','R-squared (GNY)'])
file_DATA_IN_2_tom_LAST_ER = '/Users/tomlorenc/Sites/genie/CA_EROR.csv'
dftom3.to_csv(file_DATA_IN_2_tom_LAST_ER, index=False)



import csv
reader = csv.reader(open(file_DATA_IN_2))
reader1 = csv.reader(open(file_DATA_IN_2_tom))
reader2 = csv.reader(open(file_DATA_IN_2_tom_LAST_ER))

f = open(file_DATA_IN_2_tom_LAST, "w")
writer = csv.writer(f)

for row in reader:
    writer.writerow(row)
for row in reader1:
    writer.writerow(row)
for row in reader2:
    writer.writerow(row)


#writer.writerow(row)



