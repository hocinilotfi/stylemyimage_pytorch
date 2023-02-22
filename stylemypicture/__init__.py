from flask import Flask
app = app = Flask(__name__)

app.config['SECRET_KEY'] = 'YourSecretKey'
from stylemypicture import views
