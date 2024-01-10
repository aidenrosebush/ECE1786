# -*- coding: utf-8 -*-
"""A2P3_2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10B0mndMWE5e9jrFu8Omcvm78684dpNiM
"""

from sklearn.model_selection import train_test_split
import csv

data_list = []
positive_examples = 0.0
psentences = [] #positive example sentences
plabels = []
nsentences = [] #negative example sentences
nlabels = []

with open('data.tsv') as data:
  data_file = csv.reader(data, delimiter="\t")
  i = 0 #ignore the first line
  for line in data_file:
    if i > 0:
      data_list += [line]
      data_list[i-1][1] = float(data_list[i-1][1])
      if data_list[i-1][1] == 1.0:
        psentences += [data_list[i-1][0]]
        plabels += [1.0]
      else:
        nsentences += [data_list[i-1][0]]
        nlabels += [0.0]
      positive_examples += data_list[i-1][1]
    i += 1

print(positive_examples) #find out how many examples are positive
print(i-1)

ptrain, ptestval, ptrain_labels, ptestval_labels = train_test_split(psentences, plabels, test_size = 0.36) #positive examples
ntrain, ntestval, ntrain_labels, ntestval_labels = train_test_split(nsentences, nlabels, test_size = 0.36) #negative examples

ptest, pval, ptest_labels, pval_labels = train_test_split(ptestval, ptestval_labels, test_size = 0.16/0.36) #positive examples
ntest, nval, ntest_labels, nval_labels = train_test_split(ntestval, ntestval_labels, test_size = 0.16/0.36) #negative examples

#merge
train = ptrain + ntrain
train_labels = ptrain_labels + ntrain_labels

test = ptest + ntest
test_labels = ptest_labels + ntest_labels

val = pval + nval
val_labels = pval_labels + nval_labels

overfit = ptrain[:25] + ntrain[:25]
overfit_labels = ptrain_labels[:25] + ntrain_labels[:25]

with open('train.tsv', 'w') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  for i in range(len(train)):
    tsv_writer.writerow([train[i], train_labels[i]])

with open('test.tsv', 'w') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  for i in range(len(test)):
    tsv_writer.writerow([test[i], test_labels[i]])

with open('val.tsv', 'w') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  for i in range(len(val)):
    tsv_writer.writerow([val[i], val_labels[i]])

with open('overfit.tsv', 'w') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  for i in range(len(overfit)):
    tsv_writer.writerow([overfit[i], overfit_labels[i]])

#run checks: show half the labels are 1 and half are 0. We trust sklearn matches labels to training examples.
print(sum(train_labels)/len(train_labels))
print(sum(test_labels)/len(test_labels))
print(sum(val_labels)/len(val_labels))
print(sum(overfit_labels)/len(overfit_labels))

#now check that no sample is in more than one set of data
for sentence in train:
  for i in range(0, len(val)):
    if sentence == val[i]:
      print("Repeated sentence")
  for i in range(0, len(test)):
    if sentence == test[i]:
      print("Repeated sentence")
for sentence in val:
  for i in range(0, len(test)):
    if sentence == test[i]:
      print("Repeated sentence")
#runs in about 3 seconds and shows that val and test contain nothing from train, and that test contains nothing from val.