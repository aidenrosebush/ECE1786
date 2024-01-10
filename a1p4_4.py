# -*- coding: utf-8 -*-
"""A1P4_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zpE6IOUSuRbpnoZeIU4AQPxgDea1zmvk
"""

def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    #X: samples
    #Y: labels
    #T: tokenized lemmas

    for word in textlist: #tokenize the lemmas
      T += [w2i[word]]

    X = []

    for i in range(0, len(T), 1): #generate pairs of words within the window, from part 3
      if i < (window-1)/2:
        for j in range(0, i + int((window-1)/2)+ 1): #case where the window can't extend enough to the left (ie window is 5 and we are considering the first or second word)
          if j != i:
            X += [[T[i], T[j]]]
            Y += [1]

            X += [[T[i], T[np.random.randint(0,len(T))]]] #add random pair
            Y += [-1]
      elif len(T) - 1 - i < (window-1)/2: #case where we are considering a word closer to the end than the window length on one side
        for j in range(i-int((window-1)/2), len(T)):
          if j != i:
            X += [[T[i], T[j]]]
            Y += [1]

            X += [[T[i], T[np.random.randint(0,len(T))]]] #add random pair
            Y += [-1]
      else:  #middle of the list of words, full window
        for j in range(i - int((window-1)/2), i + int((window-1)/2) + 1):
          if j != i:
            X += [[T[i], T[j]]]
            Y += [1]

            X += [[T[i], T[np.random.randint(0,len(T))]]] #add random pair
            Y += [-1]

    X = torch.tensor(X) #convert from lists to tensors for convenience of debugging
    T = torch.tensor(T)
    Y = torch.tensor(Y)

    return X, T, Y