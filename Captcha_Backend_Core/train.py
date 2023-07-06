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
import json

async def train(data_dir,learning_rate, epochs, batch_size, val_size, model_type, labId):
	"""
	Train model

	Parameters:
	-----------
	data_dir: 	str,
				Training data directory.

	learning_rate: 	float,
					Learning rate for training model.
	epochs:	int,
			Number of training epochs.

	batch_size: int,
				Batch size of training data.
	val_size: 	float,
				Size of validation set over training dataset
	model_type: string,
				Type of rnn cells for building model

	labId:	string,
			ID of lab (use for backend)
	Returns:
	--------
	Trained models saved by .h5 file
	"""

	model_path = f"./modelDir/{labId}/log_train/{model_type}"
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# Get list of all the images
	images = sorted(list(map(str, list(glob.glob(data_dir + "/*.png")))))
	labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
	characters = set(char for label in labels for char in label)
	characters = sorted(list(characters))

	print("Number of images found: ", len(images))
	print("Number of labels found: ", len(labels))
	print("Number of unique characters: ", len(characters))
	print("Characters present: ", characters)

	# Batch size for training and validation

	# Desired image dimensions
	img_width = 200
	img_height = 50

	# Factor by which the image is going to be downsampled
	# by the convolutional blocks. We will be using two
	# convolution blocks and each block will have
	# a pooling layer which downsample the features by a factor of 2.
	# Hence total downsampling factor would be 4.
	downsample_factor = 4

	# Maximum length of any captcha in the dataset
	max_length = max([len(label) for label in labels])

	# Mapping characters to integers
	char_to_num = layers.StringLookup(
		vocabulary=list(characters), mask_token=None
	)

	# Mapping integers back to original characters
	num_to_char = layers.StringLookup(
		vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
	)

	# Splitting data into training and validation sets
	x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels), 1 - val_size)

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

	def build_model(img_width, img_height):
		# Inputs to the model
		input_img = layers.Input(
			shape=(img_width, img_height, 1), name="image", dtype="float32"
		)
		labels = layers.Input(name="label", shape=(None,), dtype="float32")

		# First conv block
		x = layers.Conv2D(
			32,
			(3, 3),
			activation="relu",
			kernel_initializer="he_normal",
			padding="same",
			name="Conv1",
		)(input_img)
		x = layers.MaxPooling2D((2, 2), name="pool1")(x)

		# Second conv block
		x = layers.Conv2D(
			64,
			(3, 3),
			activation="relu",
			kernel_initializer="he_normal",
			padding="same",
			name="Conv2",
		)(x)
		x = layers.MaxPooling2D((2, 2), name="pool2")(x)

		# We have used two max pool with pool size and strides 2.
		# Hence, downsampled feature maps are 4x smaller. The number of
		# filters in the last layer is 64. Reshape accordingly before
		# passing the output to the RNN part of the model
		new_shape = ((img_width // 4), (img_height // 4) * 64)
		x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
		x = layers.Dense(64, activation="relu", name="dense1")(x)
		x = layers.Dropout(0.2)(x)

		# RNNs
		x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
		x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

		# Output layer
		x = layers.Dense(
			len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
		)(x)

		# Add CTC layer for calculating CTC loss at each step
		output = CTCLayer(name="ctc_loss")(labels, x)

		print(output)
		print(x)
		# Define the model
		model = keras.models.Model(
			inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
		)
		# Optimizer
		opt = keras.optimizers.Adam(lr=learning_rate)
		# Compile the model and return
		# model.compile(optimizer=o/pt)
		model.compile(optimizer=opt, metrics=[sequence_accuracy, character_accuracy])
		# model.compile(optimizer=opt, metrics=["accuracy"])

		return model

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = (
		train_dataset.map(
			encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
		)
		.batch(batch_size)
		.prefetch(buffer_size=tf.data.AUTOTUNE)
	)

	validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
	validation_dataset = (
		validation_dataset.map(
			encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
		)
		.batch(batch_size)
		.prefetch(buffer_size=tf.data.AUTOTUNE)
	)

	# Get the model
	model = build_model(img_width, img_height)
	model.summary()
	early_stopping_patience = 10
	# Add early stopping
	early_stopping = keras.callbacks.EarlyStopping(
		monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
	)

	# Define a callback to calculate and print accuracy
	# class AccuracyCallback(keras.callbacks.Callback):
	#     def on_epoch_end(self, epoch, logs=None):
	#         train_acc = self.model.evaluate(train_dataset, verbose=0)
	#         val_acc = self.model.evaluate(validation_dataset, verbose=0)
	#         print(f"\t train cccuracy: {train_acc} - validation accuracy:  {val_acc}")
	class LossCallback(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs=None):
			train_loss = logs["loss"]
			val_loss = logs["val_loss"]
			print(f"Train Loss (Epoch {epoch+1}): {train_loss:.4f}")
			print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")
	
	loss_callback = LossCallback()
	# Train the model
	history = model.fit(
		train_dataset,
		batch_size=batch_size,
		validation_data=validation_dataset,
		epochs=epochs,
		callbacks=[early_stopping, loss_callback]
	)

	model.save(model_path + '/model.h5')
	return {
			"history": json.dumps(history.history)
		  } 
		