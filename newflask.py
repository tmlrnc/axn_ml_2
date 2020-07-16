from flask import Flask
from collections import Counter
import requests
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_file
import subprocess
from subprocess import PIPE, Popen

UPLOAD_FOLDER = '/app/'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    <h1>Automated ML</h1>
    </br>
    </br>
    <h1>Step 1 : Upload CSV DATA File</h1>
    <h2>http://3.23.20.59/get_data </h2>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    </br>
    </br>
    <h1>Step 2: Run ML Predictions - Automated</h1>
    <h2>http://3.23.20.59/run_pred_batch </h2>
     <form method="get" action="/run_pred_batch">
    </br>
    <button type="submit">Run Predictions Automated</button>
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


@app.route('/get_covid')
def get_covid():
    path = "./ml/covid_example.csv"
    return send_file(path, as_attachment=True)

@app.route('/get_pred')
def get_pred():
    path = "./tableau-file_out_master.csv"
    return send_file(path, as_attachment=True)






if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
