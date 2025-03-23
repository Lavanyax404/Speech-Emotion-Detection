import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, LeakyReLU, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform
from keras.regularizers import l1
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.preprocess import generate_features
from src.dataload import load_data
from src.generic_utils import plot_traintest
import keras.utils as np_utils
import numpy as np
from sklearn.model_selection import train_test_split

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def give_model(lstm_units=256, learning_rate=0.0001):
    """Function to define the model architecture"""
    model = Sequential()

    model.add(LSTM(lstm_units, recurrent_dropout=0.2, input_shape=(315, 221), kernel_initializer=glorot_uniform(), return_sequences=True))
    model.add(Conv1D(512, 10, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(1024, 10, padding="same"))
    model.add(LeakyReLU())
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(lstm_units, recurrent_dropout=0.2))
    model.add(Dense(2048, activation="relu", kernel_regularizer=l1(0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_regularizer=l1(0.0005), kernel_initializer=glorot_uniform(), activation="relu"))
    model.add(Dense(12, activation="softmax"))  # Assuming 12 classes for classification
    return model

def train_model(dataset=None, model_path="./model/model.keras", labels=None, n_splits=5, learning_rate=0.0001, epochs=30, batch_size=64, verbose=True):
    """Function to train the model with K-Fold Cross-Validation"""
    
    # Setup the K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    outs = np_utils.to_categorical(labels)  # One-hot encoding the labels
    hist = {}  # History dictionary to store the results of each fold

    # Reduce Learning Rate on Plateau
    rlr = ReduceLROnPlateau(factor=0.9, verbose=1, patience=8, monitor="val_loss")
    model = give_model(lstm_units=256, learning_rate=learning_rate)  # Initialize the model
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["acc"])

    # Define callbacks to stop training early and save the best model
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)

    if verbose:
        print("[INFO] Training!")
        print(model.summary())

    # K-Fold Cross-Validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, labels)):
        if verbose:
            print(f"Shape of Fold {fold + 1}: (Train: {len(train_idx)}, Validation: {len(val_idx)})")
        
        # Preparing training and validation data
        Xtrain = generate_features(dataset[train_idx], datatype="train")
        ytrain = outs[train_idx]
        Xtest = generate_features(dataset[val_idx], datatype="valid")
        ytest = outs[val_idx]

        # Load the model only for the first fold (if you want to continue training)
        if fold == 0:
            # Compile the model only once
            model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["acc"])
        else:
            # Load the previously saved best model if it's not the first fold
            model = load_model(model_path)

        # Training the model on the current fold
        h = model.fit(Xtrain, ytrain,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose,
                      validation_data=(Xtest, ytest),
                      callbacks=[rlr, early_stop, model_checkpoint])

        # Save the model after each fold
        model.save(model_path)

        # Storing the history of the fold
        hist[fold] = h.history

    # Plot the training and validation curves
    if verbose:
        plot_traintest(hist)

def create_model(lstm_units=256, learning_rate=0.0001):
    """Wrap model creation for use with RandomizedSearchCV"""
    model = give_model(lstm_units=lstm_units, learning_rate=learning_rate)
    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["acc"])
    return model

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    from keras.wrappers.scikit_learn import KerasClassifier

    model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=64, verbose=0)
    
    param_dist = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'lstm_units': [128, 256, 512]
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    print("Best parameters found: ", random_search.best_params_)

    return random_search.best_estimator_

# Example usage
if __name__ == "__main__":
    # Load the data
    dataset, labels = load_data()  # Assuming load_data loads dataset and labels
    
    # Hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    best_model = hyperparameter_tuning(X_train, y_train)

    # Training the model with K-Fold Cross-Validation
    train_model(dataset=dataset, model_path="./model/model.keras", labels=labels, n_splits=5, learning_rate=0.0001, epochs=30, batch_size=64)
