from flask import Flask
from models import db
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

POSTGRES = {
    'user': 'postgres',
    'pw': 'genie_pass',
    'db': 'postgres',
    'host': 'localhost',
    'port': '5432',
}

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES

engine = create_engine('postgresql://postgres:genie_pass@localhost:5432/postgres')
db.init_app(app)

@app.route("/")
def main():
    return 'I am the Genie BRAIN! BRAINS BRAINS!!!!!!'

if __name__ == '__main__':
    app.run()