# fashion_search_engine
Implementation of Fashion seach engine using Triplet Neural Network on tensorflow

## Motivation:
Imagine having collection of million dataset of unkown number of classes without metadata decribing each Image and a search query it can be text or image want to find all the similar images?
This need to implement a model in which we represent each image as some vector(latent representation) in the feature space where similar will be close than other disimilar images and we are able to rank the images based on their distances from search query.

## Dataset:
Dataset I have used for training fashion seach engine can be found [here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) .This is collection of over 44k images consisting Jacket,Tshirts,Shirt's etc.However model can be trained on larger dataset for better result and lerned embeedings in feature space.

## Problem:
Here we are learning a joint embedding of the Image and Text in the Feature space. And we are doing this By Provide a Image with it's correct description and Incorrect Description and enforcing The correct Description to be as near the Image and the incorrect description to be as far from Image.

![triplet_loss_multimodal_1](https://user-images.githubusercontent.com/87687978/172415122-fa5931af-e1c7-4284-bd68-220a1c8ebe7c.png)


## Model:
This model takes three Input as Follows:
1.  Image
2.  Correct Description
3.  Incorrect Description
We are Using ResNet Model with pretrained ImageNet weights and Bidirection GRU layer for encoding Image and text respectively.And we are training By optimizing Triplet loss which is calculated as:

**Tiplet Loss = maximum( d(i,p) + margin- d(i,n), 0.0)**
Where d is euclidean Distance between embedding 
* d(i,p) is euclidean distance b/w Image embedding and Correct Description Embedding
* d(i,n) is euclidean distance b/w Image embedding and Incorrect Description Embedding
* margin is a hyperparameter 

Triplet Loss function can have Three posibilities:
* **Easy Triplet** : When d(i,n) > margin + d(i,p) then loss is zero these are called easy triplet because negative embedding is already farther in the embedding space so model will not try to waste it's time to enlarge that distance.
*  **Semi-Hard Triplets** : d(i,p) < d(i,n) < d(i,p) + margin , when negative embedding is distant than postive embedding but that distance is not greater thant the margin ,This gives postive loss.
*  **Hard Triplets**  : d(i,n) < d(i,p) when negative embedding is closer than the postive embedding from Image embedding.Thus resulting in positive loss.

![triplets_negatives](https://user-images.githubusercontent.com/87687978/172422716-b1553407-f1a6-4896-8cd9-af28b9c2b4e3.png)

> Source: https://omoindrot.github.io/triplet-loss. Representation of three “types of negatives” for an anchor and positive pair.


## Text-based Search
Top 8 search results based on the query text.These Result are based on the euclidean distance between text embedding and Image embedding in same feature Space.

![word_search6dca](https://user-images.githubusercontent.com/87687978/172433529-242909c9-b10b-4e69-8de0-722ffc792de5.jpg)
![word_searchbf1a](https://user-images.githubusercontent.com/87687978/172433573-49b4d590-7c13-44c2-a2a7-04abae94e65d.jpg)


## Image-based Search

Top 7 Search Results based on the query image.These Result are based on the euclidean distance between text embedding and Image embedding in the same feature Space.

![Image_Searchbe14](https://user-images.githubusercontent.com/87687978/172434808-afff3ef0-9afd-4881-add9-3a0e01183a60.jpg)
![Image_Search352c](https://user-images.githubusercontent.com/87687978/172434880-b2a97609-713c-455b-8341-e459ae66a7f1.jpg)


## Conclusion
In this project i have build a Image and text based seacrch engine.The basic idea is to learn meaningfull embeddings for the Image and the description in the same feature space.The important part in training was generatingh triplet  also known as triplet minning. Selecting Different kind of triplets can result in better performace ,more detailed version about triplet minning can be read [here](https://omoindrot.github.io/triplet-loss)
## References:
* https://gombru.github.io/2019/04/03/ranking_loss/
* https://omoindrot.github.io/triplet-loss
* https://towardsdatascience.com/building-a-deep-image-search-engine-using-tf-keras-6760beedbad





