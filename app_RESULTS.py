import json

from flask import Flask

app = Flask(__name__)


# We're using the new route that allows us to read a date from the URL
@app.route('/genie_results/<date>')
def genie_results(date):
    # Additionally, we're now loading the JSON file's data into file_data
    # every time a request is made to this endpoint
    with open('./data.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())

    print(file_data)

    # We can then find the data for the requested date and send it back as json
    return json.dumps(file_data)


if __name__ == '__main__':
    app.run(debug=True)