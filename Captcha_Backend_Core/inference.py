import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tool import *

import glob

async def infer(image_path, labId, model_type,batch_size, sample_model_dir = ''):
    #Model directory
    if sample_model_dir:
      model_dir = sample_model_dir
    else:
      model_dir = f'./modelDir/{labId}/log_train/{model_type}'
    # Get list of all the images
    images = [image_path]
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    # Batch size for training and validation
    max_length = max([len(label) for label in labels])
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
    x_test, y_test = np.array(images), np.array(labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = (
        test_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    
    # Register the custom layer
    
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
    prediction_model = keras.models.Model(
        loaded_model.get_layer(name="image").input, loaded_model.get_layer(name="dense2").output
    )
    prediction_model.summary()

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    list_pred = []
    #  Let's check results on some validation samples
    for batch in test_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            res =  {
            # 'img': img,
            'label' : orig_texts[i],
            'pred' : pred_texts[i]
            }
            list_pred.append(res)
        return list_pred