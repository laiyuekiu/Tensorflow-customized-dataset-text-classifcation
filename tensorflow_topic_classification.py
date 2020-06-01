###I run this in Google Colab

#from __future__ import absolute_import, division, print_function, unicode_literals

#try:
  # %tensorflow_version only exists in Colab.
#  !pip install -q tf-nightly
#except Exception:
#  pass
#import tensorflow as tf
# tf.enable_eager_execution()
#tf.executing_eagerly()

import tensorflow_datasets as tfds 
import os

#You may use your own text file 
DIRECTORY_URL = ['https://raw.githubusercontent.com/laiyuekiu/Tensorflow-customized-dataset-text-classifcation/master/crime.txt', 'https://raw.githubusercontent.com/laiyuekiu/Tensorflow-customized-dataset-text-classifcation/master/food.txt', 'https://raw.githubusercontent.com/laiyuekiu/Tensorflow-customized-dataset-text-classifcation/master/edu.txt', 'https://raw.githubusercontent.com/laiyuekiu/Tensorflow-customized-dataset-text-classifcation/master/tech.txt']
FILE_NAMES = ['crime.txt', 'food.txt', 'edu.txt', 'tech.txt']
text_dir = []
parent_dir = []

for n in range(len(FILE_NAMES)):
  text_dir.append(tf.keras.utils.get_file(FILE_NAMES[n], origin=DIRECTORY_URL[n]))

parent_dir = os.path.dirname(text_dir[0])


def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i in range(len(FILE_NAMES)):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, FILE_NAMES[i]))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

#You may adjust the neural network 
BUFFER_SIZE = 3000
BATCH_SIZE = 20
TAKE_SIZE = 200

all_labeled_data = labeled_data_sets[0]

for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(20):
  print(ex)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set, oov_buckets=3)

#This will print out the data in tensorflow format
# example_text = next(iter(all_labeled_data))[0].numpy()
# print(example_text)
# encoded_example = encoder.encode(example_text)
# print(encoded_example)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

#This will print out a sample from your neural network
# sample_text, sample_labels = next(iter(test_data))
# print('Sample text: ', sample_text[0], sample_labels[0])

vocab_size += 1

#The classification model setting
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(FILE_NAMES))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, epochs=8,
                    validation_data=test_data, 
                    validation_steps=25)


test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

#Topic classifcation prediction
def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 128)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

#This is my testing sample of topic classification
sample_pred_text = ('We are having lunch in a Chinese restaurant and we order tea and coffee. The food in there is very yummy. We are full and not feel hungry.') 
predictions = sample_predict(sample_pred_text, pad=False)
print("NO padding text: ", predictions)
predictions = sample_predict(sample_pred_text, pad=True)
print("With padding text: ", predictions)
