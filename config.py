#Here we will store all the Path and Important things  files

#Image path
image_path='myntradataset/images/'
#Caption path
caption_path='myntradataset/captions/'

#Image_size=
Image_size=(150,150,3)

#Latent vector dimension for triplet loss
vector_dims=20
#Provide the model weights path
model_weights='files/final_trained.h5'

#Provide the image repr path

img_repr='files/representation_images.csv'

#Provide the mapping json file path

vocab_mapping='files/mapping.json'

#nearest neigbour pickle file

nearest_neigbour='files/nn_model.pkl'