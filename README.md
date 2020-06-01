The tensorflow_topic_classification.py is a sample of using Tensorflow RNN text classification with own data.

I use tf.data.TextLineDataset to load my customized text file into tensorflow.

tf.data.TextLineDataset will read your text file line by line and then import into tensorflow line by line.

crime.txt, food.txt, edu.txt and tech.txt are the sample text files to train the Tensorflow RNN classification.
The training result is the classification can identify the input is related to which topics.

All the sample texts are from online resources and I don't own any of them. 
