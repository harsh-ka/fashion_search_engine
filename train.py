import tensorflow as tf
import glob
import os
import json
from keras.callbacks import ModelCheckpoint
from config import image_path,caption_path,model_weights
from preprocessing import image_loading,text_loading
import tensorflow.keras import layers
from models import base_model

#loading the captions and the images paths
test_size=.1
images_list=sorted(glob.glob(os.path.join(image_path,'*.jpg')))
caption_list=sorted(glob.glob(os.path.join(caption_path,'*.txt')))

train_images_list=images_list[:-int(len(images_list)*test_size)]
train_caption_list=caption_list[:-int(len(caption_list)*test_size)]

test_images_list=images_list[-int(len(images_list)*test_size):]
test_caption_list=caption_list[-int(len(caption_list)*test_size):]

print(f"Total training Images and Captions are {len(train_images_list),len(train_caption_list)} \
        and Total test Images and captions are {len(test_images_list),len(test_caption_list)}")


#Loading the trained data and Test data
#We are not loading the entire as the entire data is huge so loading it batch by batch for training

'''Dataset will contain a Image know as Anchor tag and a positive_label/caption which belongs to anchor tag 
and a negative label/caption which doesn't belongs to the anchor tag'''

batch_size=64
train_images=tf.data.Dataset.from_tensor_slices((train_images_list)).map(image_loading)
train_postive_caption=tf.data.TextLineDataset(train_caption_list).map(text_loading)
train_negative_caption=tf.data.TextLineDataset(train_caption_list).shuffle(1024).map(text_loading)

train_data=tf.data.Dataset.zip((train_images,train_postive_caption,train_negative_caption)).batch(batch_size,drop_remainder=True).prefetch(32)


test_images=tf.data.Dataset.from_tensor_slices((test_images_list)).map(image_loading)
test_postive_caption=tf.data.TextLineDataset(test_caption_list).map(text_loading)
test_negative_caption=tf.data.TextLineDataset(test_caption_list).shuffle(1024).map(text_loading)

test_data=tf.data.Dataset.zip((test_images,test_postive_caption,test_negative_caption)).batch(batch_size,drop_remainder=True).prefetch(32)


# Text vectorizer layer it adapt on a vocbulary and convert input sentences in number encoded vector
# Ex "This is me"---------->[1,11,5] it will look like this
if not os.path.isfile('mapping.json'):

    vectorizer = layers.TextVectorization(max_tokens=100000, output_sequence_length=10, output_mode='int')

    vectorizer.adapt(train_postive_caption.batch(batch_size))

    word_mapping = dict(zip(vectorizer.get_vocabulary(), range(len(vectorizer.get_vocabulary()))))

    json.dump(vectorizer.get_vocabulary(), open('mapping.json', 'w+'))
else:

    mapping = json.load(open('mapping.json', 'r+'))
    # Intilaizing the vector layer for text processing for the embedding layer

    vectorizer = layers.TextVectorization(max_tokens=100000, output_mode='int', output_sequence_length=10)

    # Loading the vocabulary saved in the mapping.json file

    vectorizer.set_vocabulary(mapping)

    word_mapping = dict(zip(vectorizer.get_vocabulary(), range(len(vectorizer.get_vocabulary()))))

    print(vectorizer.get_vocabulary())

def vectorize_text(Images,postive_caption,negative_caption):
  #print(Images.shape[0])
  return ({'images':Images,\
          'positive_input':vectorizer(postive_caption),\
          'negative_input':vectorizer(negative_caption)
          },tf.random.uniform(shape=(batch_size,1)))

train_data=train_data.map(vectorize_text).repeat()
test_data=test_data.map(vectorize_text).repeat()

model=base_model(vec_dims=vector_dims,vocab_size=len(word_mapping),)


#Steps per epoch calculation
train_per_epochs=len(train_images_list)//batch_size
test_per_epochs=len(test_images_list)//batch_size

#model checkpoint
file_path='/content/training_weights.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,save_weights_only=True, mode='min'\
                             ,save_best_only=True)

history=model.fit(train_data,steps_per_epoch=train_per_epochs,validation_steps=test_per_epochs,\
          validation_data=test_data,epochs=30,callbacks=[checkpoint])

def freeze_layers(model):
    for i in model.layers:
        i.trainable = True
        if isinstance(i, Model):
            freeze_layers(i)
    return model

model=freeze_layers(model)
model.save_weights(model_weights)