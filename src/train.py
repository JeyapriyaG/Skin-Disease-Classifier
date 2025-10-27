import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
import pandas as pd

def train(train_csv, val_csv, images_dir, epochs=10, batch_size=16):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    valgen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(train_df, directory=images_dir, x_col='image_id', y_col='dx',
                                            target_size=(224,224), class_mode='categorical', batch_size=batch_size)
    val_gen = valgen.flow_from_dataframe(val_df, directory=images_dir, x_col='image_id', y_col='dx',
                                         target_size=(224,224), class_mode='categorical', batch_size=batch_size)

    model = build_model(num_classes=train_gen.num_classes)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('../models/best_model.h5', monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save('../models/final_model.h5')
    print("Training complete!")
