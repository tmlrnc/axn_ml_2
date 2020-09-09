"""
API

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
