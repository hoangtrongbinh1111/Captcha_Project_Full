import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tool import *

import glob

async def test(test_data_dir, labId, model_type, batch_size, sample_model_dir = ''):
    """
    Testing trained models.

    Parameters:
    ----------
    test_data_dir:	string,
    				Directory path of testing data.

    labId: 	string,
    		ID of lab.

    model_path: 	int,
    				Path model for testing.

    model_type: 	string,
    				Type of rnn cell model.

    Returns:
    --------
    Accuracy of testing model on the testing dataset.

    """

    test_acc = 0
    
    #Model directory
    if sample_model_dir:
      model_dir = sample_model_dir
    else:
      model_dir = f'./modelDir/{labId}/log_train/{model_type}'


    # Get list of all the images
    images = sorted(list(map(str, list(glob.glob(test_data_dir + "/*.png")))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]


    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    # Desired image dimensions
    img_width = 200
    img_height = 50
    characters = ['2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'm',
    'n',
    'p',
    'w',
    'x',
    'y']
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )
    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
      vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    def sequence_accuracy(y_true, y_pred):
      y_pred = decode_batch_predictions(y_pred)
      for label in y_true:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)
      batch_size = y_true.shape[0]
      total_sequences = y_true.shape[1]
      correct_sequences = 0

      for i in range(batch_size):
        if np.array_equal(orig_texts[i], y_pred[i]):
          correct_sequences += 1

      sequence_accuracy = correct_sequences / batch_size
      return sequence_accuracy

    def character_accuracy(y_true, y_pred):
      total_characters = np.sum(y_true != -1)
      correct_characters = np.sum(y_true == y_pred)

      character_accuracy = correct_characters / total_characters
      return character_accuracy
    # Define the custom layer
    # Register the custom layer
    # Register the custom layer
    keras.utils.get_custom_objects()["CTCLayer"] = CTCLayer
    loaded_model = keras.models.load_model(model_dir + '/model.h5', custom_objects={"sequence_accuracy": sequence_accuracy, "character_accuracy": character_accuracy })
    downsample_factor = 4
    def encode_single_sample(img_path, label):
      # 1. Read image
      img = tf.io.read_file(img_path)
      # 2. Decode and convert to grayscale
      img = tf.io.decode_png(img, channels=1)
      # 3. Convert to float32 in [0, 1] range
      img = tf.image.convert_image_dtype(img, tf.float32)
      # 4. Resize to the desired size
      img = tf.image.resize(img, [img_height, img_width])
      # 5. Transpose the image because we want the time
      # dimension to correspond to the width of the image.
      img = tf.transpose(img, perm=[1, 0, 2])
      # 6. Map the characters in label to numbers
      label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
      # 7. Return a dict as our model is expecting two inputs
      return {"image": img, "label": label}

    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in labels])
    x_test, y_test = np.array(images), np.array(labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (
        test_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    evaluation_results = loaded_model.evaluate(test_dataset, verbose=1)

    #yield for backend.

    return {
            "evaluation_results": evaluation_results
          } 