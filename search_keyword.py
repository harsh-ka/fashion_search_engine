import sklearn
import json
import os
from ast import literal_eval
import pandas as pd
from config import img_repr,nearest_neigbour,vocab_mapping,model_weights
import pickle5 as pickle
import matplotlib.pyplot as plt
from models import text_model
from preprocessing import image_loading
import argparse
from tensorflow import keras
import numpy as np
from uuid import uuid4
parser=argparse.ArgumentParser()
parser.add_argument('-w','--word',type=str,help='Enter the word you want to search image for', \
                    required=True
                    )
args=parser.parse_args()


word=args.word

vocabulary=json.load(open(vocab_mapping,'r'))

t_model=text_model(vocab_size=len(vocabulary))
t_model.load_weights(model_weights,by_name=True)


data=pd.read_csv(img_repr,header=None,names=['image_list','captions','image_embedding','text_embedding'],\
                 converters={'image_embedding':literal_eval})
nn=pickle.load(open(nearest_neigbour,'rb'))

print(word)
def search_by_word(word, t_model, nearest_neigbour, vectorizer, file_name=False):
    # i_model should be an instance of Model
    assert isinstance(nearest_neigbour, sklearn.neighbors.NearestNeighbors)
    # t_model should be an instance of Model

    assert isinstance(t_model, keras.Model)
    # vectorzer should be instance of Textvectorizer

    assert isinstance(vectorizer, keras.layers.TextVectorization)

    if not vectorizer.is_adapted:
        if os.path.isfile(vocab_mapping):
            vocabulary = json.load(open(vocab_mapping, 'r'))
            vectorizer.set_vocabulary(vocabulary)
        else:
            raise FileExistsError('Please created a mapping.json in this directory')
    #print(vectorizer(np.array(['mens shirt'])))
    label_word = vectorizer(np.array([word])).numpy()
    text_repr = t_model.predict(label_word)
    # now we have text repr

    # Now we will use the nearest neigbour to obtain the near search queries
    indices = nearest_neigbour.kneighbors(text_repr, n_neighbors=8, return_distance=False)
    # We are getting 8 neighbours
    images = data.loc[indices[0]]['image_list'].tolist()
    plt.figure(figsize=(10, 15),)
    plt.suptitle('Query text :{}'.format(word),fontsize='x-large')
    for i in range(len(indices[0])):
        plt.subplot(4, 2, i + 1)
        plt.imshow(image_loading(images[i][1:], False, [300, 300]))
        plt.title("Result image %s" % i)
        plt.xticks([])
        plt.yticks([])
    if file_name :
        plt.savefig('word_search%s.jpg'%str(uuid4())[:4], dpi=200)

    plt.show()


search_by_word(str(word),t_model,nn,keras.layers.TextVectorization(output_sequence_length=10),True)