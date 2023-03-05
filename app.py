from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image  
import PIL  


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.reshape = tf.keras.layers.Reshape((8, 8, 256))
        self.convT2 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.convT3 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()   
        self.convT4 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()         
        self.convT5 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.convT2(x)
        x = self.batchnorm2(x, training=training)
        x = self.relu(x)
        x = self.convT3(x)
        x = self.batchnorm3(x, training=training)
        x = self.relu(x)    
        x = self.convT4(x)
        x = self.batchnorm4(x, training=training)
        x = self.relu(x)         
        x = self.convT5(x)
        return x

    
   

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate():
    # call your GAN model and generate an image
    # Load the trained GAN model
    model = Generator()
    model.load_weights('./Models/generator_weights') 
    noise=tf.random.normal([1, 100])
    image=(model(noise))
    fake_image_array = np.array(image[0])
    
    # Create PIL Image object from NumPy array
    fake_image_pil = Image.fromarray(fake_image_array,'RGB')
    # save the image to a file
    image_path = './static/images/generated_image.png'  
    # Save PIL Image object to file
    fake_image_pil.save(image_path)    

    # render the HTML page with the image tag
    return render_template('generated_image.html', image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)

