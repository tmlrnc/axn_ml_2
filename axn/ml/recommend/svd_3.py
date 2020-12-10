
import pandas as pd

file_in_name2 = '/Users/tomlorenc/Downloads/retail_sales.csv'
file_out_name_retail = '/Users/tomlorenc/Downloads/top_retail_sales_predictions_new.csv'
file_out_name_location = '/Users/tomlorenc/Downloads/top_retail_locations_new.csv'
file_out_name_fraud = '/Users/tomlorenc/Downloads/top_fraud_predictions_new.csv'
df1 = pd.read_csv(file_in_name2)

fraud = df1[df1['Amount'] >= 100.0]
df2 = df1.drop(fraud.index)
print(df2)

fraud.to_csv(file_out_name_fraud, index=False)
df3 = df2.groupby(['ProductName', 'StoreLocationStreet'])['Amount'].agg('sum')
mystore = df2['StoreLocationStreet'].unique().tolist()
mylen = len(mystore)
print('*********')
df4 = df2.groupby(['ProductName'])['Amount'].sum().reset_index()

df5 = df4.sort_values(by=['Amount'], ascending=False)
dfObj1 = df5[['ProductName', 'Amount']].head(5)

dfObj1['Amount'] = dfObj1['Amount'].apply(lambda x: int(x/mylen))
dfObj1.to_csv(file_out_name_retail, index=False)

import numpy as np

mylatlong2 = np.unique(df2[['Latitude', 'Longitude']], axis=0)

print('*********mylatlong2')
print(mylatlong2)

from sklearn.cluster import KMeans
id_n=4
kmeans = KMeans(n_clusters=id_n, random_state=0).fit(mylatlong2)
id_label=kmeans.labels_

mycluster = kmeans.cluster_centers_
pd.DataFrame(mycluster).to_csv(file_out_name_location, index=False)


