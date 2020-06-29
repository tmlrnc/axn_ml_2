from flask import Flask, jsonify
from flask import request

app = Flask(__name__)

@app.route('/genie_read', methods=['GET', 'POST', 'DELETE', 'PUT'])
def add():
    data = request.get_json()
    # ... do your business logic, and return some response
    # e.g. below we're just echo-ing back the received JSON data
    return jsonify(data)




if __name__ == '__main__':
    app.run(debug=True)