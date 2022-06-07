def base_model(vec_dims=20, vocab_size=100000, embedding_matrix=None):
    # checking if embedding_matrix is of correct shape or not if provided

    if embedding_matrix is not None:
        assert embedding_matrix.shape == (vocab_size, 100)

    Image_input = layers.Input(shape=(150, 150, 3), name='images')
    Text_label_1 = layers.Input(shape=(None,), name='positive_input')
    Text_label_2 = layers.Input(shape=(None,), name='negative_input')
    image_extractor = keras.applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    #
    '''
    image_extractor.trainable=False
    '''

    '''Here we are not training the Entire Resnet model We are just training last few layers for fine tuning '''
    for layer in image_extractor.layers[:]:
        if 'conv5_block' in layer.name or 'conv4_block5' in layer.name or 'conv4_block4' in layer.name:
            layer.trainable = True
            # print(layer.name)
        else:
            layer.trainable = False

    # print(image_extractor.otuput.shape)
    # Passing the Image through Resnet model than applying global max pooling and connecting to Dense layer to
    # produce output vector
    x1 = image_extractor(Image_input)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Image output Layer
    Image_output = layers.Dense(vec_dims, activation='linear', name='dense_image_1')(x1)

    # Normalisation layer for processing
    norm_layer = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))

    # Embedding layer to Convert the given text in vector of shape 50
    # Shape Embedding layer expects is (Batch_size,word_length) and gives output (Batch_size,word_length,50)

    embed = layers.Embedding(vocab_size, 50, name="embed", input_length=100, embeddings_initializer='uniform' \
        if embedding_matrix is None else keras.initializers.Constant(embedding_matrix))

    # Using bi direction GRU layer as it performs better that unidirectional layer

    gru = layers.Bidirectional(layers.GRU(256, return_sequences=True), name="gru_1")
    # Finally defining a Dense layer for output of text layer
    dense_2 = layers.Dense(vec_dims, activation="linear", name="dense_text_1")

    x2 = embed(Text_label_1)
    x2 = layers.SpatialDropout1D(0.1)(x2)
    x2 = gru(x2)

    # Might be eliminated on the future words
    x2 = layers.GlobalMaxPool1D()(x2)
    Text_label_output_1 = dense_2(x2)

    x3 = embed(Text_label_2)
    x3 = layers.SpatialDropout1D(0.1)(x3)
    x3 = gru(x3)
    x3 = layers.GlobalMaxPool1D()(x3)
    Text_label_output_2 = dense_2(x3)

    # normlaising the layer for loss calcultion
    Image_output = norm_layer(Image_output)
    Text_label_output_1 = norm_layer(Text_label_output_1)
    Text_label_output_2 = norm_layer(Text_label_output_2)

    output = layers.Concatenate(axis=-1)([Image_output, Text_label_output_1, Text_label_output_2])

    model = Model(inputs=[Image_input, Text_label_1, Text_label_2], outputs=output)

    model.compile(loss=triplet_loss, optimizer=keras.optimizers.Adam(lr=.001))
    return model

'''We are Defining a Image model which will Process the Image to prodcut the output vector and 
 whose architecture is Same as the Architecture used for training base model(Very important),
 as we will be using the same trained weight of base model for this Image model for prediction on the Image'''


def image_model(vec_dim=20,lr=0.0001):
    #Image Input Layer
    input_1 = layers.Input(shape=(150,150,3))
    #Resnet Model Image extraction
    image_extractor = keras.applications.ResNet50(weights='imagenet', include_top=False,input_shape=(150,150,3))

    x1 = image_extractor(input_1)
    x1 = layers.GlobalMaxPool2D()(x1)
    #Dense layer for output vector
    dense_1 = layers.Dense(vec_dim, activation="linear", name="dense_image_1")

    x1 = dense_1(x1)
    #Normalizing the output
    _norm = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))

    x1 = _norm(x1)

    model = Model([input_1], x1)

    model.compile(loss="mae", optimizer=keras.optimizers.Adam(lr))
    return model

def text_model(vocab_size=100000,vec_dim=20,lr=0.0001):
    input_2 = layers.Input(shape=(None,))

    embed = layers.Embedding(vocab_size, 50, name="embed")
    gru = layers.Bidirectional(layers.GRU(256, return_sequences=True), name="gru_1")
    dense_2 = layers.Dense(vec_dim, activation="linear", name="dense_text_1")

    x2 = embed(input_2)
    x2 = gru(x2)
    x2 = layers.GlobalMaxPool1D()(x2)
    x2 = dense_2(x2)

    _norm = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))

    x2 = _norm(x2)

    model = Model([input_2], x2)

    model.compile(loss="mae", optimizer=keras.optimizers.Adam(lr))
    return model