# -*- coding: utf-8 -*-
"""A1P3_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15EeFyKz0A9Xezy4mDVd00I49mDfVbvwZ
"""

def train_word2vec(textlist, window, embedding_size ):
    loss = torch.nn.CrossEntropyLoss()

    batchsize = 4
    epochs = 50

    # Set up a model with Skip-gram (predict context with word)
    # textlist: a list of the strings

    # Create the training data

    pairs = tokenize_and_preprocess_text(textlist, v2i, window)

    # Split the training data

    train = pairs[:int(0.8*len(pairs))]
    val = pairs[int(0.8*len(pairs)):]

    train_losses = []
    val_losses = []
    batchlist = []

    # instantiate the network & set up the optimizer

    w = Word2vecModel(11,2)

    optimizer = torch.optim.Adam([w.embedding], lr=0.0005)

    # training loop

    for epoch in range(epochs):
      tloss = 0.0 #average training loss per epoch
      vloss = 0.0 #average validation loss per epoch
      for i in range(int(np.ceil(len(train)/batchsize))):
        targets = [] #get target words in the batch
        answers = [] #get answer words in the batch
        #may be a way to vectorize this but heres another for loop for now
        for j in range(min(batchsize, len(train) - i*batchsize)):
          targets += [v2i[train[batchsize*i + j][0]]] #get indices of target words
          answers += [v2i[train[batchsize*i + j][1]]] #get indices of correct answers
        #convert targerts and answer indices to tensors
        tt = torch.tensor(targets)
        ta = torch.tensor(answers)
        optimizer.zero_grad()
        logits = w.forward(tt)
        l = loss(logits, ta)
        l.backward()
        optimizer.step()
        tloss += l.item()/(len(train)) #increment avg loss
      #compute average loss on validation set
      for i in range(len(val)):
        vtarget = v2i[val[i][0]] #get indices of target words
        vanswer = v2i[val[j][1]] #get indices of correct answers
        vtt = torch.tensor(vtarget)
        vta = torch.tensor(vanswer)
        logits = w(vtt)
        l = loss(logits, vta)
        vloss += l.item()/(len(train)) #increment avg loss
      train_losses += [float(tloss)]
      val_losses += [float(vloss)]

    epochs = np.linspace(0,epochs-1,epochs)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label = 'Training loss')
    ax.plot(epochs, val_losses, label = 'Validation loss')
    plt.title('Losses by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')

    return w