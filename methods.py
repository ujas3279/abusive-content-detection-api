from keras import utils
import json
import run
from flask import abort
from logging.config import dictConfig
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#utility
import re
from keras.models import load_model
import tensorflow as tf
df = pd.read_csv("sentiment_data.csv")
tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['tweet_content'].values)
def run_model(img):
	try :
	   model = tf.keras.models.load_model('model.h5')
       
	except FileNotFoundError as e :        
	   return abort('Unable to find the file: %s.' % str(e), 503)
	pred = model.predict(img)
	prediction = pred[0][0]
	if(prediction>=0.38):
		status=1
	else:
		status=0
	return status
def load_image(filename):
	
	x = tokenizer.texts_to_sequences(filename)
	x = pad_sequences(x, maxlen=20, padding='post')
	return x
def classify(data):
	upload = data
	image = load_image(upload)
	#load_image() is to process image :
	print('image ready')
	try:
		prediction = run_model(image)
		return (json.dumps({"prediction": str(prediction)}))
	except FileNotFoundError as e:
		return abort('Unable to locate image: %s.' % str(e), 503)
	except Exception as e:
		return abort('Unable to process image: %s.' % str(e), 500)




