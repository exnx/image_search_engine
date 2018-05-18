# image_search_engine

This program was used for a Kaggle challenge to create the highest accurate match of a text description and a corresponding photo.  I was part of a 3 person team where each person implemented a different algorithm, and at the
end we performed an ensemble optimization to achieve better results.  The following is my implementation using word2vec.

# Kaggle challenge link:
https://www.kaggle.com/c/cs5785-fall-2017-final

Link to report: https://docs.google.com/document/d/1jCBU04vJFyKMTYyQnxHtOfy7W5GXWU3ixlg4a0HM48k/edit#heading=h.uwvtm4rfn2wr

Files:

word2vec.ipynb - contains implementation photo search engine using of word2vec
setup.py - a module used to retrieve information from the training and test data (omitted in this repo due to size)


This program uses word2vec to match text descriptions to the closest photo matching that description. Both the training and test photos are tagged with keywords and descriptions.

We are given 10,000 training photos, each with 5 sentences describing the photo. We are also given 2,000 test photos, also with 5 sentences describing the photo.

word2vec is a neural network model from Google that maps words to their most common surrounding words, allowing you incorporate the context, or order of words, in your prediction. We will be using the gensim library to access the word2vec model, and train it with our own data.

From a high level, word2vec works by taking a large corpus of sentences, and understands the context of words in sentences. It maps a word to other commonly associated words. For example, "dog" might commonly occur with "pet", "animal" or even "cat".

After training with a corpus, a high dimensional space is created. This allows you to take a word vector and project it into this high dimensional space. What you might observe is that words that commonly occur together will project to the same region or cluster near each other in this "space".

For our purposes, we take a given test sentence, averaged all the word vectors, and then found its nearest neighbor word2vec based on cosine similarity distance, in the training data. We then mapped the nearest 20 neighbor descriptions in the test data and used their photo labels as the predictions to submit.
