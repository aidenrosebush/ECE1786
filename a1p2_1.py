# -*- coding: utf-8 -*-
"""A1P2_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YkNDxgPpyGre1XL0izXcf1wEO9e5FFTd
"""

#import torch
#import torchtext

#Imports just to be sure if you want to run this manually with just this file
#glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              #dim=50)    # embedding size = 50

#assume that we're passing the complete category list of words to the function. Could use print_closest_cosine_words to create the
#category, but the question doesn't ask for this explicitly
def compare_words_to_category(category, target):
  category_glove = []
  for word in category:
    category_glove += [glove[word].numpy()] #assemble list of embeddings for words in the category
  category_glove = torch.tensor(category_glove) #convert to tensor
  target_glove = glove[target]
  c_size = len(category) #how many words in the category
  avg_cosine = (1/c_size)*sum(torch.cosine_similarity(category_glove, target_glove.unsqueeze(0)))

  print(target+" average cosine distance: "+str(avg_cosine))

  avg_embedding = (1/c_size)*sum(category_glove) #average out embeddings
  avg_embedding_distance = torch.cosine_similarity(avg_embedding, target_glove.unsqueeze(0)) #compute cosine similarity to average

  print(target+ " average embedding distance: "+str(avg_embedding_distance.squeeze(0)))

  return [avg_cosine, avg_embedding_distance]