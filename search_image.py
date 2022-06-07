import sklearn
import json
from ast import literal_eval
import pandas as pd
from config import img_repr,nearest_neigbour,vocab_mapping,model_weights
import pickle
import matplotlib.pyplot as plt
from models import image_model
from preprocessing import image_loading
import argparse
import tensorflow as tf
import os

parser=argparse.ArgumentParser()
parser.add_argument('-i','--image_path',type=str,help='Enter the image_path you want to search image for', \
                    required=True
                    )
args=parser.parse_args()

image_path=args.image_path

vocabulary=json.load(open(vocab_mapping,'r'))

i_model=image_model()
i_model.load_weights(model_weights,by_name=True)


data=pd.read_csv(img_repr,header=None,names=['image_list','captions','image_embedding','text_embedding'],\
                 converters={'image_embedding':literal_eval})
nn=pickle.load(open(nearest_neigbour,'rb'))

def search_by_image(image_path, i_model, nearest_neigbour, file_name=None):
    assert isinstance(nearest_neigbour, sklearn.neighbors.NearestNeighbors)
    # query Image shape should be equal to that of i_model.input

    image = image_loading(image_path, True)

    assert image.shape == i_model.inputs[0].shape[-3:]

    img_embedding = i_model.predict(tf.expand_dims(image, axis=0))

    # After getting the Image embedding i.e (batch_size,embedding_size)

    idx = nearest_neigbour.kneighbors(img_embedding, n_neighbors=8, return_distance=False)

    similar_image_path = data.iloc[idx[0]]['image_list'].tolist()

    # ploting Query and Similar Image
    plt.figure(figsize=(20, 20))

    for i in range(len(similar_image_path)):
        plt.subplot(2, 4, i + 1)
        if i == 0:
            plt.imshow(image_loading(image_path, False, [300, 300]))
            plt.title('Query Image')
        else:
            plt.imshow(image_loading(similar_image_path[i][1:], False, [300, 300]))
            plt.title('Similary image %s' % (i))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    if file_name is not None:
        plt.savefig(str(file_name), dpi=200)
    plt.show()


search_by_image(str(image_path),i_model,nn,)