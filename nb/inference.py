
#'pip install tflite-runtime'

#import tflite_runtime as tflite
import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
from PIL import Image
import numpy as np

from keras_image_helper import create_preprocessor

def preprocessing_input(x):
    x /= 127.5
    x -= 1.
    return x

path = "../test/shorts/78d2c5d2-b19d-422a-aff0-a6eb78291b4f.jpg"

def load_and_process_img(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.NEAREST)

    X = np.array(img, dtype = 'float32')
    X = np.array([X])
    X = preprocessing_input(X)
    return X


#loading model
Interpreter = tflite.Interpreter("fashion_model.tflite")
#loading the weight from the model to memory
Interpreter.allocate_tensors()

input_index = Interpreter.get_input_details()[0]['index']
output_index = Interpreter.get_output_details()[0]['index']

classes = [
    'dress',
     'hat',
     'longsleeve',
     'outwear',
     'pants',
     'shirt',
     'shoes',
     'shorts',
     'skirt',
     't-shirt',]

def predict(path):
    X = load_and_process_img(path)

    Interpreter.set_tensor(input_index, X)
    Interpreter.invoke()
    preds = Interpreter.get_tensor(output_index)

    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    
    url = event['url']
    results = predict(url)
    return results





#print(predict(path))




