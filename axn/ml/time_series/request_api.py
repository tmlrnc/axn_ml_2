"""
One Hot Encoder Module

<img src="images/ohe.png" alt="OHE">

Machine learning algorithms cannot work with categorical data directly.
Categorical data must be converted to numbers.
This module encodes categorical features as a one-hot numeric array.
A one hot encoding is a representation of categorical variables as binary vectors.
This first requires that the categorical values be mapped to integer values.
Then, each integer value is represented as a binary vector that is all zero values
except the index of the integer, which is marked with a 1.

The input to this module should be a csv file where each column feature is a category as a string like
"United States, Canada" or "Male, Female"

Then we take the string and convert to array of integers
denoting the values taken on by categorical (discrete) features.

Like
"United States, Canada" becomes (1,2)
"Male, Female" becomes (3,4)

The features are encoded using a one-hot aka ‘one-of-K’ or ‘dummy’ encoding scheme.
This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)


Like
"United States, Canada" becomes (1,2) that encodes to (001, 010)
"Male, Female" becomes (3,4) that encodes to (100, 101)


By default, the encoder derives the categories based on the unique values in each feature.
Alternatively, you can also specify the categories manually.

This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.


Parameters
----------
file_in: file
    raw csv file input to be predicted. Must be a csv file where first row has column header
file_out: file
    csv file output encoded using one-hot one-of-K encoding scheme
ignore: string
    columns of data to NOT be encoded


Returns:
----------
    csv file output encoded using one-hot one-of-K encoding scheme.


Example 1. CSV Files:
---------------------
python -m ohe  \


  --file_in csvs/sales.csv \




  --file_out_ohe csvs/sales_ohe.csv \



  --ignore id


Example 1 - Data Input CSV File:
----------------------------
<img src="images/sales.png" alt="OHE" width="600" height="300">


Example 1 - One Hot Encoded CSV File:
-----------------------------
<img src="images/sales_ohe.png" alt="OHE" width="600" height="300">






Example 2:
----------
python -m ohe  \


  --file_in csvs/states.csv \




  --file_out_ohe csvs/states_ohe.csv \



  --ignore none



Example 2 Data Input File:
----------
ALASKA

ARIZONA

ARKANSAS

CALIFORNIA

COLORADO

CONNECTICUT

DELAWARE

FLORIDA

GEORGIA


Example 2 OHE Output File:
----------



1	0	0	0	0	0	0	0	0	0


0	1	0	0	0	0	0	0	0	0


0	0	1	0	0	0	0	0	0	0


0	0	0	1	0	0	0	0	0	0


0	0	0	0	1	0	0	0	0	0


0	0	0	0	0	1	0	0	0	0


0	0	0	0	0	0	1	0	0	0


0	0	0	0	0	0	0	1	0	0


0	0	0	0	0	0	0	0	1	0


0	0	0	0	0	0	0	0	0	1




Example 3:
----------
python -m ohe  \


  --file_in csvs/gender.csv \


  --file_out_ohe csvs/gender_ohe.csv \



  --ignore none




Example 3 Data Input File:
----------
MALE

MALE

FEMALE

FEMALE

MALE

FEMALE

FEMALE

MALE

FEMALE

FEMALE



Example 3 OHE Output File:
----------


0	1

0	1

1	0

1	0

0	1

1	0

1	0

0	1

1	0

1	0



Example 4:
----------
python -m ohe  \


  --file_in csvs/pets.csv \


  --file_out_ohe csvs/pets_ohe.csv \



  --ignore none


Example 4 Data Input File:
----------
DOG

CAT

CAT

NONE

DOG

DOG

NONE

NONE

CAT

DOG




Example 4 OHE Output File:
----------

0	1	0

1	0	0

1	0	0

0	0	1

0	1	0

0	1	0

0	0	1

0	0	1

1	0	0

0	1	0




"""

import requests
# api-endpoint

API_UPLOAD = "http://3.23.20.59:5000/upload_csv"
# sending get request and saving the response as response object
print("*******************************************")
req = requests.get(API_UPLOAD)
print("API_UPLOAD")
print("*******************************************")


API_RUN = "http://3.23.20.59:5000/run_pred"
# sending get request and saving the response as response object
print("*******************************************")
req = requests.get(API_RUN)
print("API_RUN")
print("*******************************************")



API_GET = "http://3.23.20.59:5000/get_pred"
# sending get request and saving the response as response object
req = requests.get(API_GET)
print("NEXT TOP SALES PREDICTED FOR TOMORROW")
print("*******************************************")
print(str(req.text))
