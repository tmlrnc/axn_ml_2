from flask import Flask
from collections import Counter
import requests
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_file
import subprocess
from subprocess import PIPE, Popen

#UPLOAD_FOLDER = '/home/ubuntu/flaskapp'
UPLOAD_FOLDER = '/app/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/myupfiles_run_gny_ml', methods=['GET', 'POST'])
def myupfiles_run_gny_ml():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     <h1>Step 4 : Download Predictions CSV</h1>
      <form method="get" action="/get_pred">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Get Predictions</button>
     </form>
     </br>
     '''


@app.route('/myupfiles_run_demo_data_retail', methods=['GET', 'POST'])
def myupfiles_run_demo_data_retail():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''

@app.route('/myupfiles_run_demo_data_fraud', methods=['GET', 'POST'])
def myupfiles_run_demo_data_fraud():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''

@app.route('/myupfiles_run_demo_data_location', methods=['GET', 'POST'])
def myupfiles_run_demo_data_location():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''


@app.route('/myupfiles_fraud2', methods=['GET', 'POST'])
def myupfiles_fraud2():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''


@app.route('/myupfiles_fraud', methods=['GET', 'POST'])
def myupfiles_fraud():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run Machine Learning in your GNY Web Wallet </h1> 
     </div> 
</br>
     <h2>You can now access powerful ML securely through your GNY Wallet in three different ways.  You can 
 <ul>
  <li>run GNY ML Demos</li>
  <li>run plug-n-play ML contracts (new contracts added quarterly)</li>
  <li>build custom GNY ML contracts with Jupyter Notebooks</li>
</ul> 
 </h2>
    </br>
    </br>
     <h1>run GNY ML Demos </h1>
    </br>
      <form method="get" action="/run_demo">
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Demos</button>
     </form>
     </br>
    </br>
     <h1>run plug-n-play ML contracts</h1>
    </br>
      <form method="get" action="/run_demo_data">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run GNY ML with your Data</button>
     </form>
     </br>
    </br>
     <h1>build custom GNY ML contracts with Jupyter Notebooks </h1>
    </br>
      <form method="get" action="/run_jupiter">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Customize GNY ML with Jupyter</button>
     </form>
     </br>
    '''


@app.route('/myupfiles_location', methods=['GET', 'POST'])
def myupfiles_location():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run Machine Learning in your GNY Web Wallet </h1> 
     </div> 
</br>
     <h2>You can now access powerful ML securely through your GNY Wallet in three different ways.  You can 
 <ul>
  <li>run GNY ML Demos</li>
  <li>run plug-n-play ML contracts (new contracts added quarterly)</li>
  <li>build custom GNY ML contracts with Jupyter Notebooks</li>
</ul> 
 </h2>
    </br>
    </br>
     <h1>run GNY ML Demos </h1>
    </br>
      <form method="get" action="/run_demo">
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Demos</button>
     </form>
     </br>
    </br>
     <h1>run plug-n-play ML contracts</h1>
    </br>
      <form method="get" action="/run_demo_data">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run GNY ML with your Data</button>
     </form>
     </br>
    </br>
     <h1>build custom GNY ML contracts with Jupyter Notebooks </h1>
    </br>
      <form method="get" action="/run_jupiter">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Customize GNY ML with Jupyter</button>
     </form>
     </br>
    '''



@app.route('/myupfiles', methods=['GET', 'POST'])
def myupfiles():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run Machine Learning in your GNY Web Wallet </h1> 
     </div> 
</br>
     <h2>You can now access powerful ML securely through your GNY Wallet in three different ways.  You can 
 <ul>
  <li>run GNY ML Demos</li>
  <li>run plug-n-play ML contracts (new contracts added quarterly)</li>
  <li>build custom GNY ML contracts with Jupyter Notebooks</li>
</ul> 
 </h2>
    </br>
    </br>
     <h1>run GNY ML Demos </h1>
    </br>
      <form method="get" action="/run_demo">
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Demos</button>
     </form>
     </br>
    </br>
     <h1>run plug-n-play ML contracts</h1>
    </br>
      <form method="get" action="/run_demo_data">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run GNY ML with your Data</button>
     </form>
     </br>
    </br>
     <h1>build custom GNY ML contracts with Jupyter Notebooks </h1>
    </br>
      <form method="get" action="/run_jupiter">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Customize GNY ML with Jupyter</button>
     </form>
     </br>
    '''


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run Machine Learning in your GNY Web Wallet </h1> 
     </div> 
</br>
     <h2>You can now access powerful ML securely through your GNY Wallet in three different ways.  You can 
 <ul>
  <li>run GNY ML Demos</li>
  <li>run plug-n-play ML contracts (new contracts added quarterly)</li>
  <li>build custom GNY ML contracts with Jupyter Notebooks</li>
</ul> 
 </h2>
    </br>
    </br>
     <h1>run GNY ML Demos </h1>
    </br>
      <form method="get" action="/run_demo">
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Demos</button>
     </form>
     </br>
    </br>
     <h1>run plug-n-play ML contracts</h1>
    </br>
      <form method="get" action="/run_demo_data">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run GNY ML with your Data</button>
     </form>
     </br>
    </br>
     <h1>build custom GNY ML contracts with Jupyter Notebooks </h1>
    </br>
      <form method="get" action="/run_jupiter">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Customize GNY ML with Jupyter</button>
     </form>
     </br>
    '''



@app.route('/run_jupiter')
def run_jupiter():
    print('end2')
    return '''
     <!doctype html>
     <title></title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
     <h1>Customize GNY ML with Jupyter</h1>
     </div> 
<h2>
CUSTOMIZE YOUR GNY ML CONTRACTS
</h2>
<h3>
</br>
GNY is the world’s first decentralised machine learning platform catering to developers and enterprises. We use blockchain technology to deliver a secure, collaborative platform for machine learning, data capture and analysis. 

</br>
</br>
We provide our users with hundreds of machine learning functions and contracts securely on chain (including data preparation). However, sometimes a user may wish to customize those contracts, or build them from scratch to better suit their needs. To allow users this flexibility we have integrated Jupyter Notebooks into the GNY WebWallet. 

</br>
</br>
Jupyter Notebooks boast the ability to allow its users create and share documents that contain live code, equations, visualizations and narrative text. So users of the GNY platform will now have the option to begin with our plug-n-play contracts, modify them if needed, or even build contracts totally from scratch while still accessing the full power of our GNY Brain ML engine that incorporates:


</br>
</br>
The GNY Data Preparation API which includes the ability to:
</br>

    Rescale data.
</br>

    Standardize data.
</br>

    Normalize data.
</br>

    Binarize data.

</br>
    One Hot encode data
</br>

    Feature select data
</br>
</br>

GNY's exclusive data prep tech transforms your raw data, allowing your selected machine learning algorithm to reveal incredibly accurate insights, predictions, and results.

</br>

Our GNY Brain ML engine also incorporates:

</br>
The GNY Machine Learning Prediction API with 100 algorithms including:

</br>
    Regressions
</br>

    Classification
</br>

    Neural Nets

</br>
    Clustering
</br>

    Deep Learning

</br>
</br>
Our proprietary software helps select which algorithms provide the best correlations/ predictions results, and help deploy those results to guide future actions.
For this we have integrated Jupyter Notebooks into the GNY WebWallet. You can start with on our plug-n-play contracts and modify it, or build contracts from scratch while still accessing the full power of our GNY Brain ML engine.  
</br>

</br>
</br>
Sign In to JupyterHub using your own username/password
</br>
</br>
You will see your JupyterHub admin screen
</br>
</br>
Click on New Terminal 
</br>
</br>
You will see your New Terminal screen
</br>
</br>
Enter the command: python gny_retail_ml/gny_retail_ml.py
</br>
</br>
You will see the GNY Retail Machine Learning Code run:
</br>
</br>
Enter the command: python gny_retail_ml/gny_retail_ml.py
</br>
</br>
You will see you editing GNY Retail Machine Learning Code 

</br>

</h3>
</br>
      <form method="get" action="/get_jup_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML with Jupyter User Guide</button>
     </form>
</br>
</br>

</br>
 <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;"  onclick="window.location.href='http://18.219.197.224/hub/login';">
Customize GNY ML with Jupyter </button>
     </br>
     '''

@app.route('/get_jup')
def get_jup():
    print('end2')
    return '''
     <!doctype html>
     <title>COMING SOON</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
     <h1>Customize GNY ML with Jupyter</h1>
     </div> 

     </br>
     '''


@app.route('/run_gny_ml')
def run_gny_ml():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run GNY ML with your Data </h1>
     </div> 
    </br>
    </br>
    <h1>Step 1 : Upload YOUR CSV DATA File</h1>
    <form method=post action="/myupfiles_run_gny_ml" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 2: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''

@app.route('/run_demo_data_retail')
def run_demo_data_retail():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>RETAIL DEMO</h1>
     </div> 
    </br>
    <h1>Step 1 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles_run_demo_data_retail" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 2: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''

@app.route('/run_demo_location')
def run_demo_location():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>LOCATION DEMO</h1>
     </div> 
     <h2>Predict locations of customers </h2>
    </br>
    <h3>
The location of your customers using a multiobjective DBSCAN spatial clustering algorithm to find the optimal clusters using the spatial data collected in the sales area. 
    </br>
We do Spatial Data Mining with a K-means algorithm that starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids. It halts creating and optimizing clusters when the centroids have stabilized — there is no change in their values because the clustering has been successful.
</h3>
    </br>
     <h1>Step 1 : Download Example CSV</h1>
      <form method="get" action="/get_covid">
     <button type="submit">Download Example CSV</button>
     </form>
     </br>
    </br>
    <h1>Step 2 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles_run_demo_data_location" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_loc">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
	</br>
      <form method="get" action="/get_loc_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML Tech Notes</button>
     </form>
</br>
     '''


@app.route('/run_demo_retail_2')
def run_demo_retail_2():
    print('i here')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>RETAIL DEMO</h1>
     </div> 
     <h2>Predict what will be my next top selling item  </h2>
    </br>
    <h3>A sales prediction model for retail stores using the deep learning neural net  given the sales of a particular day for one year, predicts the top sales for any following day. 
</br>
This is a deep learning model that considers the L1 regularization achieved sales forecasting accuracy rate of 86%. 
</br>
The products at the retail store have been finely categorized. With classification our genie deep learning is able to establish correlations between a persons and a product's attributes. 
</br>
Genie Deep Learning maps these input attributes to outputs. It finds correlations. 
</br>
It is known as a universal approximator, because it can learn to approximate an unknown function f(x) = y between any input x and any output y, assuming they are related at all by correlation or causation, for example. 
</br>
In the process of learning, a neural network finds the right f, or the correct manner of transforming x into y. </h3>
</br>
     <h1>Step 1 : Download Example CSV</h1>
      <form method="get" action="/get_covid">
     <button type="submit">Download Example CSV</button>
     </form>
     </br>
    <h1>Step 2 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
	</br>
	</br>
      <form method="get" action="/get_retail_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML Tech Notes</button>
     </form>
</br>
</br>

     '''



@app.route('/run_demo_retail')
def run_demo_retail():
    print('i here')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>RETAIL DEMO</h1>
     </div> 
     <h2>Predict what will be my next top selling item  </h2>
    </br>
    <h3>A sales prediction model for retail stores using the deep learning neural net  given the sales of a particular day for one year, predicts the top sales for any following day. 
</br>
This is a deep learning model that considers the L1 regularization achieved sales forecasting accuracy rate of 86%. 
</br>
The products at the retail store have been finely categorized. With classification our genie deep learning is able to establish correlations between a persons and a product's attributes. 
</br>
Genie Deep Learning maps these input attributes to outputs. It finds correlations. 
</br>
It is known as a universal approximator, because it can learn to approximate an unknown function f(x) = y between any input x and any output y, assuming they are related at all by correlation or causation, for example. 
</br>
In the process of learning, a neural network finds the right f, or the correct manner of transforming x into y. </h3>
</br>
     <h1>Step 1 : Download Example CSV</h1>
      <form method="get" action="/get_covid">
     <button type="submit">Download Example CSV</button>
     </form>
     </br>
    <h1>Step 2 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_retail">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
	</br>
	</br>
	</br>
      <form method="get" action="/get_retail_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML Tech Notes</button>
     </form>
</br>
     '''



@app.route('/run_demo_data_location')
def run_demo_data_location():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>LOCATION DEMO</h1>
     </div> 
     <h2>Predict locations of customers </h2>
    </br>
    <h3>
The location of your customers using a multiobjective DBSCAN spatial clustering algorithm to find the optimal clusters using the spatial data collected in the sales area. 
    </br>
We do Spatial Data Mining with a K-means algorithm that starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids. It halts creating and optimizing clusters when the centroids have stabilized — there is no change in their values because the clustering has been successful.
</h3>
    </br>
    <h1>Step 1 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles_run_demo_data_location" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 2: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_loc">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''


@app.route('/run_demo_data_fraud')
def run_demo_data_fraud():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds with your data </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
    </br>
    <h1>Step 1: Upload YOUR CSV DATA File</h1>
    <form method=post action="/myupfiles_run_demo_data_fraud" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 2: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_fraud">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
     '''

@app.route('/run_demo_fraud2')
def run_demo_fraud2():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
     <h1>Step 1 : Download Example CSV</h1>
      <form method="get" action="/get_covid">
     <button type="submit">Download Example CSV</button>
     </form>
     </br>
    <h1>Step 2 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles_fraud" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_fraud">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
	</br>
	</br>
      <form method="get" action="/get_fraud_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML Tech Notes</button>
     </form>
</br>
     '''


@app.route('/run_demo_fraud')
def run_demo_fraud():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>FRAUD DEMO</h1>
     </div> 
     <h2>Predict what transactions are frauds  </h2>
    </br>
    <h3>
Credit card fraud using a neural net trained to know a person type and amount and frequency of transactions. Because of confidentiality issues, we cannot provide the original features and more background information about the data.
    </br>
Therefore we transform all sales into features ‘Time’ and ‘Amount’. Feature ‘Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature ‘Amount’ is the transaction, These features can be used for neural net learning. 
    </br>
Feature ‘Class’ is the response variable and it takes value 1 in case of fraud and 0 otherwise. Given the class imbalance ratio, we measure the accuracy using the Area Under the Precision-Recall Curve. 
</h3>
    </br>
     <h1>Step 1 : Download Example CSV</h1>
      <form method="get" action="/get_covid">
     <button type="submit">Download Example CSV</button>
     </form>
     </br>
    <h1>Step 2 : Upload CSV DATA File</h1>
    <form method=post action="/myupfiles_fraud2" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 3: Run ML Predictions - Automated</h1>
     <form method="get" action="/run_pred_batch_2_fraud">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Predictions</button>
    </form>
    </br>
     </br>
	</br>
      <form method="get" action="/get_fraud_ins">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Get GNY ML Tech Notes</button>
     </form>
</br>
     '''


@app.route('/run_demo_data')
def run_demo_data():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>RUN GNY ML WITH YOUR DATA</h1>
     </div> 
    </br>
    </br>
	<h1>RUN GNY ML WITH YOUR DATA - RETAIL </h1>
     <h2>Predict what will be my next top selling item  </h2>
     <form method="get" action="/run_demo_data_retail">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Retail Demo</button>
    </form>
    </br>
     <h1>RUN GNY ML WITH YOUR DATA - LOCATION </h1>
     <h2>Predict what locations will sell the most inventory next month </h2>
      <form method="get" action="/run_demo_data_location">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run Location Demo</button>
     </form>
     </br>
    </br>
     <h1>RUN GNY ML WITH YOUR DATA - FRAUD </h1>
     <h2>Predict what transactions are frauds  </h2>
      <form method="get" action="/run_demo_data_fraud">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Run Fraud Demo</button>
     </form>
     </br>
     '''



@app.route('/run_demo')
def run_demo():
    print('demo')
    return '''
    <!doctype html>
    <title>Upload CVS DATA File</title>
     <div style="background-color:#5C89A8;color:white;padding:20px;">
    <h1>Run GNY ML Demos</h1>
     </div> 
    </br>
    <h2>Learn how to run ML contracts using your own data by practicing with our Demos. 
    </br>
Each Demo shows the kind of data required, and the steps taken to successfully implement 3 different retail focused ML contracts.</h2>
    </br>
    </br>
    <h1>RETAIL DEMO</h1>
     <h2>Predict what will be my next top selling item  </h2>
     <form method="get" action="/run_demo_retail">
    </br>
	     <button style="color:white;background-color:#409EFF;height:32px;width:125px;border-radius: 12px;" type="submit">Run Retail Demo</button>
    </form>
    </br>
     <h1>LOCATION DEMO</h1>
     <h2>Predict what locations will sell the most inventory next month </h2>
      <form method="get" action="/run_demo_location">
	     <button style="color:white;background-color:#67C23A;height:32px;width:125px;border-radius: 12px;" type="submit">Run Location Demo</button>
     </form>
     </br>
    </br>
     <h1>FRAUD DEMO</h1>
     <h2>Predict what transactions are frauds  </h2>
      <form method="get" action="/run_demo_fraud">
	     <button style="color:white;background-color:red;height:32px;width:125px;border-radius: 12px;" type="submit">Run Fraud Demo</button>
     </form>
     </br>
     '''


@app.route('/run_pred_batch_2_fraud')
def run_pred_batch_2_fraud():
    my_env = os.environ.copy()
    my_env["PARENT_DIR"] = "/app/"
    p = subprocess.Popen(['bash',  'test_retail.sh'], stdin=PIPE,stdout=PIPE, env=my_env)

    one_line_output = p.stdout.read()
    print('end2')
    print(one_line_output)
    return '''
     <!doctype html>
     <title>DOWNLOAD PREDICTIONS</title>
     <h1>Automated ML</h1>

     <h1>Step 4 : Download New Predictions CSV</h1>
      <form method="get" action="/get_pred_3_fraud">
     <button type="submit">Get New Predictions</button>
     </form>
     </br>
     '''


@app.route('/run_pred_batch_2_loc')
def run_pred_batch_2_loc():
    my_env = os.environ.copy()
    my_env["PARENT_DIR"] = "/app/"
    p = subprocess.Popen(['bash',  'test_retail.sh'], stdin=PIPE,stdout=PIPE, env=my_env)

    one_line_output = p.stdout.read()
    print('end2')
    print(one_line_output)
    return '''
     <!doctype html>
     <title>DOWNLOAD PREDICTIONS</title>
     <h1>Automated ML</h1>

     <h1>Step 4 : Download New Predictions CSV</h1>
      <form method="get" action="/get_pred_3_loc">
     <button type="submit">Get New Predictions</button>
     </form>
     </br>
     '''

@app.route('/run_pred_batch_2_retail')
def run_pred_batch_2_retail():
    my_env = os.environ.copy()
    my_env["PARENT_DIR"] = "/app/"
    p = subprocess.Popen(['bash',  'test_retail.sh'], stdin=PIPE,stdout=PIPE, env=my_env)

    one_line_output = p.stdout.read()
    print('end2')
    print(one_line_output)
    return '''
     <!doctype html>
     <title>DOWNLOAD PREDICTIONS</title>
     <h1>Automated ML</h1>

     <h1>Step 4 : Download New Predictions CSV</h1>
      <form method="get" action="/get_pred_3_retail">
     <button type="submit">Get New Predictions</button>
     </form>
     </br>
     '''


@app.route('/run_pred_batch')
def run_pred_batch():
    my_env = os.environ.copy()
    my_env["PARENT_DIR"] = "/app/"
    p = subprocess.Popen(['bash',  'test_covid_automate_24.sh'], stdin=PIPE,stdout=PIPE, env=my_env)

    one_line_output = p.stdout.read()
    print('end2')
    print(one_line_output)
    return '''
     <!doctype html>
     <title>DOWNLOAD PREDICTIONS</title>
     <h1>Automated ML</h1>

     <h1>Step 3 : Download Predictions CSV</h1>
     <h2>http://3.23.20.59/get_pred </h2>
      <form method="get" action="/get_pred">
     <button type="submit">Get Predictions</button>
     </form>
     </br>
     '''


@app.route('/get_fraud_ins')
def get_fraud_ins():
    path = "./Genie_FraudNotes.pdf"
    return send_file(path, as_attachment=True)


@app.route('/get_retail_ins')
def get_retail_ins():
    path = "./Genie_RetailNotes.pdf"
    return send_file(path, as_attachment=True)


@app.route('/get_loc_ins')
def get_loc_ins():
    path = "./Genie_LocationNotes.pdf"
    return send_file(path, as_attachment=True)


@app.route('/get_jup_ins')
def get_jup_ins():
    path = "./GNYJupyterHub.pdf"
    return send_file(path, as_attachment=True)

@app.route('/get_covid')
def get_covid():
    path = "./retail_sales.csv"
    return send_file(path, as_attachment=True)

@app.route('/upload_csv')
def upload_csv():
    path = "./top_retail_sales_predictions.csv"
    return 

@app.route('/run_pred')
def run_pred():
    path = "./top_retail_sales_predictions.csv"
    return 


@app.route('/get_pred_3_loc')
def get_pred_3_loc():
    path = "./top_retail_locations_new.csv"
    return send_file(path, as_attachment=True)

@app.route('/get_pred_3_fraud')
def get_pred_3_fraud():
    path = "./top_fraud_predictions_new.csv"
    return send_file(path, as_attachment=True)

@app.route('/get_pred_3_retail')
def get_pred_3_retail():
    path = "./top_retail_sales_predictions_new.csv"
    return send_file(path, as_attachment=True)

@app.route('/get_pred')
def get_pred():
    path = "./top_retail_sales_predictions.csv"
    return send_file(path, as_attachment=True)


@app.route('/get_loc_pred')
def get_loc_pred():
    path = "./top_retail_locations.csv"
    return send_file(path, as_attachment=True)


@app.route('/get_fraud_pred')
def get_fraud_pred():
    path = "./top_fraud_predictions.csv"
    return send_file(path, as_attachment=True)






if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000,debug=True)
