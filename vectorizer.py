import os
import json

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