from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation,MaxPooling2D,Dropout,BatchNormalization,AveragePooling2D,Input,ZeroPadding2D,Add
def convblock(X, f, filters):
    # Filter
    F1, F2, F3 = filters
    # Save the input value
    X_shortcut = X

    # MAIN PATH
    # First
    X = Conv2D(F1, (1, 1), padding='valid')(X)
    X = Dropout(0.3)(X)

    X = Activation('relu')(X)

    # Second
    X = X = Conv2D(F2, (f, f), padding='same')(X)
    X = Dropout(0.3)(X)

    X = Activation('relu')(X)

    # Third
    X = X = Conv2D(F3, (1, 1), padding='valid')(X)
    X = Dropout(0.3)(X)

    # SHORTCUT PATH
    X_shortcut = Conv2D(F3, (1, 1))(X_shortcut)
    X_shortcut = Dropout(0.3)(X_shortcut)

    # add shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def idblock(X, f, filters):
    # filters
    F1, F2, F3 = filters
    X_shortcut = X

    # First part
    X = Conv2D(F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=1))(X)
    X = Dropout(0.3)(X)

    X = Activation('relu')(X)

    # Second part
    X = Conv2D(F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=1))(X)
    X = Dropout(0.3)(X)

    X = Activation('relu')(X)

    # Third part
    X = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=1))(X)
    X = Dropout(0.3)(X)

    # Shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def myres(shape, classes):
    # Inputs
    X_input = Input(shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(16, (3, 3), strides=(2, 2))(X)
    X = Dropout(0.3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convblock(X, f=3, filters=[16, 16, 64])
    X = idblock(X, 3, [16, 16, 64])
    X = idblock(X, 3, [16, 16, 64])
    # Stage 3
    X = convblock(X, f=3, filters=[32, 32, 128])
    X = idblock(X, 3, [32, 32, 128])
    X = idblock(X, 3, [32, 32, 128])

    # AVGPOOL.
    X = AveragePooling2D((2, 2))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.6)(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='myres')

    return model