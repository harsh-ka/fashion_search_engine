#Image processing and text loading scripts
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
def image_loading(Image_path,preprocessing=True,Image_size=[150,150]):
  raw_image=tf.io.read_file(Image_path,)
  image = tf.image.decode_jpeg(raw_image,channels=3,)
  image = tf.image.convert_image_dtype(image, tf.float32,)
  image = tf.image.resize(image, Image_size,)
  #preprocessing_the_input_for resnet model
  if preprocessing:
    image=  preprocess_input(image*255)
  return image

def text_loading(text):
  return tf.strings.regex_replace(text,'[%s]'% re.escape(string.punctuation),' ')