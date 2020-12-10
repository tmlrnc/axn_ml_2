import numpy as np
import pandas as pd
import scipy
import pandas as pd
from datetime import datetime
import requests
import numpy as np


URL = 'http://3.23.20.59:5000/myupfiles_run_demo_data_location'

FILE = '/Users/tomlorenc/Sites/genie/test1/retail_sales.csv'
files = {
    'data' : data,
    'document': open('file_name.pdf', 'rb')
}
r = requests.post(URL, files=files, headers=headers)
