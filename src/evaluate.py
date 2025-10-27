import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate(model_path, test_csv, images_dir):
    model = tf.keras.models.load_model(model_path)
    test_df = pd.read_csv(test_csv)
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_dataframe(test_df, directory=images_dir, x_col='image_id', y_col='dx',
                                           target_size=(224,224), class_mode='categorical', batch_size=16, shuffle=False)
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    print(confusion_matrix(y_true, y_pred))
