
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
            plt.imshow(image_loading(similar_image_path[i], False, [300, 300]))
            plt.title('Similary image %s' % (i))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    if file_name is not None:
        plt.savefig(str(file_name), dpi=200)
    plt.show()