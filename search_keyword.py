import sklearn
import json
from ast import literal_eval
import pandas as pd
from config import img_repr,nearest_neigbour,vocab_mapping,model_weights
import pickle
import matplotlib.pyplot as plt
from models import text_model
from preprocessing import image_loading
import argparse
from tensorflow import keras
parser=argparse.ArgumentParser()
parser.add_argument('-w','--word',type='string',help='Enter the word you want to search image for', \
                    required=True
                    )
args=parser.parse_args()


word=args['word']

vocabulary=json.load(open(vocab_mapping,'r'))

t_model=text_model(vocab_size=len(vocabulary))
t_model.load_weights(model_weights,by_name=True)


data=pd.read_csv(img_repr,header=None,names=['image_list','captions','image_embedding','text_embedding'],\
                 converters={'image_embedding':literal_eval})
nn=pickle.load(open(nearest_neigbour,'rb'))

def seach_by_word(word, t_model, nearest_neigbour, vectorizer, file_name=None):
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

    label_word = vectorizer([word]).numpy()
    text_repr = t_model.predict(label_word)
    # now we have text repr

    # Now we will use the nearest neigbour to obtain the near search queries
    indices = nearest_neigbour.kneighbors(text_repr, n_neighbors=8, return_distance=False)
    # We are getting 8 neibours
    images = data.loc[indices[0]]['image_list'].tolist()
    plt.figure(figsize=(20, 20))
    for i in range(len(indices[0])):
        plt.subplot(4, 2, i + 1)
        plt.imshow(image_loading(images[i], False, [300, 300]))
        plt.title("Similar image %s" % i)
        plt.xticks([])
        plt.yticks([])
    if file_name is not None:
        plt.savefig(str(file_name), dpi=200)

    plt.show()


search_by_word(word,t_model,nn,keras.layers.TextVectorization(output_sequence_length=10),'test_query.jpg')