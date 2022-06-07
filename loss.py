import tensorflow.keras.backend as K
def triplet_loss(y_true, y_pred, margin=1.0):
    '''This is Special type of Loss function Using for this Image retrival
    In This loss function we train our model on One Ancor ,One Positive Label ,One Negative Label
    Main Idea behind This loss function is To push The Negative label away From the anchor Label in the Embedding Space and Pull the Postive Label
    Towards the Anchor label in the Embedding Space
    Loss function is : d_ap- d_an+ margin    (i.e d_ap is euclidean distance b/w anchor and postive label and d_an is euclidean b/w anchor and negative)
    Margin is One of the very important parameters :
    There can be three posbilties based on the margin values:
    d_an > margin+d_ap   (Called Easy negative ,Not very much to learn from)
    d_ap < d_an <margin  (Called Semi hard ,Usefull)
    d_an < d_ap          (Called hard tiplets,as it is close to to anchor tag than postive label and we need to push it away in the embedding space)

    '''

    # We are casting the y_pred values to float-32 to avoid any errors int future

    y_pred = K.cast(y_pred, 'float32')

    # y_pred is concatenation of [Anchor ,Postive,Negative]:
    length = y_pred.shape[-1]
    Anchor = y_pred[:, :int(length * 1 / 3)]
    Postive = y_pred[:, int(length * 1 / 3):int(length * 2 / 3)]
    Negative = y_pred[:, int(length * 2 / 3):]

    d_ap = K.sum(K.square(Anchor - Postive), axis=-1)
    d_an = K.sum(K.square(Anchor - Negative), axis=-1)

    loss = K.maximum(d_ap - d_an + margin, 0)
    return K.mean(loss)