import numpy as np
import pandas as pd
import scipy
import pandas as pd
from datetime import datetime
import requests
import numpy as np


file_in = '/Users/tomlorenc/Sites/genie/data/LA_df_first.pkl'

file_out_1 = '//Users/tomlorenc/Sites/genie/data/LA_df_corr.csv'
la_df = pd.read_pickle(file_in)

print(la_df['demand'])

cols = la_df.columns
print(cols)

print(la_df['dailycoolingdegreedays'])
#la_df['A'].corr(df['B'])


mycor_c = la_df['demand'].corr(la_df['dailycoolingdegreedays'])

print(mycor_c)

mycor_h = la_df['demand'].corr(la_df['dailyheatingdegreedays'])


print(mycor_h)

df_plot = la_df[['demand', 'dailycoolingdegreedays']]
print(df_plot)

df_plot.to_csv(file_out_1)



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

GENIE_EIA_API = '3cca91939d85a450c5a182d18020e63e'

# collect electricty data for Los Angeles
REGION_CODE = 'LDWP'
url = 'http://api.eia.gov/category/?api_key=3cca91939d85a450c5a182d18020e63e&category_id=338985'

# megawatthours
url_demand = requests.get('http://api.eia.gov/series/?api_key=%s&series_id=EBA.%s-ALL.D.H' % (GENIE_EIA_API, REGION_CODE)).json()
electricity_df = Genie_EIA_request_to_df(url_demand, 'demand')

# clean electricity_df of outlier values. this cut removes ~.01% of the data
electricity_df = electricity_df[electricity_df['demand'] != 0]

print(electricity_df)

exit()

