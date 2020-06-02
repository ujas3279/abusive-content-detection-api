from flask import Flask,Response , request , flash , url_for,jsonify
from flask import Flask
import methods
app = Flask(__name__)
@app.route('/')
def hello():
	return "Hello World!"
@app.route('/predict',methods=['POST'])
def prediction():
	upload = [request.args['data']]
	a=methods.classify(upload)
	return a

if __name__ == '__main__':
	app.run(host="0.0.0.0",port=8080)