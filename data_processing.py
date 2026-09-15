"""OCR the source PDFs and build masked-token training CSVs.

Part of RedactedGPT. Originally developed as Colab notebooks; converted to scripts.
Paths are resolved from DATA_ROOT in config.py — set REDACTEDGPT_DATA to override.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, MODEL_ROOT  # noqa: E402

# Commented out IPython magic to ensure Python compatibility.

from pdf2image import convert_from_path, convert_from_bytes #these imports for processing the images
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pytesseract as pt
import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image

import csv #these imports for creating training datasets from the cleaned_master_2.txt file
from transformers import AutoTokenizer

# Commented out IPython magic to ensure Python compatibility.
# %ls

pdfNames = ['DOC_0001229684.pdf',  'DOC_0001525482.pdf',  'DOC_0006220800.pdf',  'DOC_0006297294.pdf',  'DOC_0006364591.pdf']
#all of the scanned documents

ifl = [] #imagefilelist: list of image files to be stored here on this notebook which we then convert to .txt
for pdf in pdfNames:
  imagefile = convert_from_path(pdf)
  ifl += [imagefile]

#convert single pdf to image (when we already have the .PNG)
imagefile = convert_from_path('911 Commission Report.pdf')
ifl += [imagefile]

#convert images saved in this notebook into textfiles
for i, imagefile in enumerate(ifl):
  with open(f"{DATA_ROOT}/Textfiles/"+str(pdfNames[i][:-4]), 'w') as f:
    for image in imagefile:
      f.write(pt.image_to_string(image)) #create separate files based on the filenames; could be used for cleaning one at a time

#convert images saved in Drive to txt
basename = "911 Commission Report 101-200-"

with open(f"{DATA_ROOT}/Textfiles/commission", 'w') as c:
  for i in range(1,586,1):
    print(i)
    if i < 10:
      filename = basename + "00"+str(i) #maintaining consistent naming of the files
    elif i < 100:
      filename = basename + "0"+str(i)
    else:
      filename = basename +str(i)
    c.write(pt.image_to_string(Image.open(f"{DATA_ROOT}/Images/Commission Images/911 Commission Report Images/"+filename+'.png')))

#consolidate all textfiles into one master textfile for easy training
with open(f"{DATA_ROOT}/Textfiles/master", 'w') as master:
  for filename in pdfNames:
    with open(f"{DATA_ROOT}/Textfiles/"+filename[:-4], 'r') as f:
      master.write(f.read())
    f.close()
  with open(f"{DATA_ROOT}/Textfiles/commission", 'r') as c:
    master.write(c.read())

#clean up master file
import csv
with open(f"{DATA_ROOT}/Textfiles/master", 'r') as master:
  for i in range(40):
    line = master.readline()
    if i == 6:
      blankline = line #unsure what characters represent the blank lines, so manually found one and removed it
with open(f"{DATA_ROOT}/Textfiles/cleaned_master.txt", 'w') as cm:
  with open(f"{DATA_ROOT}/Textfiles/master", 'r') as master:
    while True:
      line = master.readline()
      if not line:
        break
      elif line != blankline:
        cm.write(line)
with open(f"{DATA_ROOT}/Textfiles/cleaned_master.txt", 'r') as cm:
  for i in range(2000):
    line = cm.readline()
    if i == 38:
      blankline = line #there seem to be several types of blank lines to remove
    elif i == 1033:
      unknown_char = line[0] #weird character keeps appearing (probably outside of unicode)
    elif i == 1796:
      unknown_char2 = line[0]
    elif i == 1034:
      blankline2 = line

with open(f"{DATA_ROOT}/Textfiles/cleaned_master_2.txt", 'w') as cm:
  with open(f"{DATA_ROOT}/Textfiles/cleaned_master.txt", 'r') as master:
    while True:
      line = master.readline()
      if not line:
        break
      #remove commonly occuring patterns from declassified documents
      #strings of capital letters only occur acround redacted sections and those lines contain nothing else
      #also removing common section titles or headers
      #these if statements were all manually added based on reading the txt flies and looking at repeated characters
      elif line != blankline and "SECR" not in line and "SEGR" not in line and "SESR" not in line and "SECG" not in line:
        if "DCI Counterterrorist Center" not in line and "Approved for Release:" not in line:
          if "August 2001" not in line and "SCOPE AND METHODOLOGY" not in line:
            if "SECRE" not in line and line[0] != unknown_char and line != blankline2 and line[0] != unknown_char2:
              if "FINDINGS" not in line:
                if "INTRODUCTION" not in line and line[:6] != "Figure":
                  if "SCOPE & METHODOLOGY" not in line and "SEE" not in line and "WORD" not in line:
                    if len(line) > 11: #many lines only have a few weird characters. worst case we cut out only a couple words
                      if line[:3] == "-- ":
                        cm.write(line[3:])
                      else:
                        cm.write(line)
# with open(f"{DATA_ROOT}/Textfiles/cleaned_master_2.txt", 'r') as cm:
#   for i in range(2500):
#     line = cm.readline()
#     print(str(i) + " "+line)

#find most frequently used words
with open(f"{DATA_ROOT}/Textfiles/cleaned_master_2.txt", 'r') as master:
  m = master.read()

worddict = {}
sorted_worddict = {}
lines = m.split("\n")
for line in lines:
  words = line.split()
  for word in words:
    if word not in worddict:
      worddict[word] = 1
    else:
      worddict[word] += 1

sorted_wordlist = sorted(worddict, key=worddict.get, reverse = True)

#possible keywords for training (may or may not be in each tokenizer's vocab)
#these were chosen arbitrarily from sorted_wordlist. Could be expanded or improved with more context on the domain
master_keywords = ['Bin', 'Ladin', 'CIA', 'FBI', 'Al', 'Qaeda', 'FAA', 'Clarke', 'interrogation', 'CTC', 'Hazmi', 'NYPD', 'Mihdhar', 'Taliban', 'Rice', 'Binalshibh', 'Abu', 'Battalion', 'Richard', 'Binalshibh,','Jarrah', 'Berger', 'Berger,','Khallad', 'DOJ', 'Khallad,','Moussaoui','Ladin.','DOD','bin','Shehhi', 'Hanjour', 'CIA,','Usama','Qaeda.','Islamist', 'Hamburg','Ahmed','Pentagon', 'Hadley', 'Omar','Abdullah','Qaeda,','William','Bayoumi','suicide','Ramzi','Rumsfeld','Yousef', 'Ali','Thomas', 'Samuel', 'Sudan', 'nuclear', 'Shelton', 'Ressam','Hazmi,','Ashcroft','Condoleezza', 'Sheikh', 'Sudanese', 'Shehhi', 'Mohamed', 'Laden', 'Jarrah','Hambali','Nawaf', 'Saeed', 'Nashiri', 'Cressey', 'Clarke,','Ghamdi']
print(len(master_keywords))

#add keywords to tokenizer vocab
#generate training data file where target tokens are only word in this new tokenizer vocabulary

def update_tokenizer_get_training_data(model_name, sorted_wordlist, master_keywords):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.add_special_tokens({"mask_token": "[MASK]"}) #add standard special tokens for consistency, so model training files are consistent
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenizer.add_special_tokens({'eos_token': '[EOS]'})
  tokenizer_keywords = [] #custom list of keywords

  initial_vocab = tokenizer.vocab
  print(len(initial_vocab))
  for word in master_keywords:
    if word not in initial_vocab:
      tokenizer.add_tokens(word)

  vocab = tokenizer.vocab
  print(len(vocab))

  #save the modified tokenizer so we may access it in the training files
  if model_name == "microsoft/deberta-base":
    model_name = "deberta" #model name is used in naming files, the / character implies a directory which doesn't exist

  if model_name == "bert-base-cased":
    tokenizer.save_pretrained(f"{DATA_ROOT}/Tokenizer Files/BERT_tokenizer")
  elif model_name == "gpt2":
    tokenizer.save_pretrained(f"{DATA_ROOT}/Tokenizer Files/gpt2_tokenizer")
  elif model_name == "deberta":
    tokenizer.save_pretrained(f"{DATA_ROOT}/Tokenizer Files/DeBERTa_tokenizer")

  #create training lines: sentences followed by the target in CSV format
  with open(f"{DATA_ROOT}/Textfiles/cleaned_master_2.txt", 'r') as master:
    with open(f"{DATA_ROOT}/CSVs/traininglines_"+model_name+'.csv', 'w') as tl:
      csvwriter = csv.writer(tl, delimiter = ',')
      linestring = ""
      while True:
        line = master.readline()
        if not line:
          break
        linestring = line
        splitline = line.split()
        for i, word in enumerate(splitline):
          if word in master_keywords:
            for k in range(i - 1, i + 1, 1): #create training examples by sampling 1 word before and 1 words after
              if k >= 0 and k < len(splitline) and splitline[k] in vocab:
                csvwriter.writerow([linestring] + [splitline[k]])
        lines = []
        linestring = ""

  tl.close()
  master.close()

  #create training data, in CSV format, by masking the label from the input sentences- so each row looks like (masked input, target)
  sep = " " #separator for joining training sentences
  with open(f"{DATA_ROOT}/CSVs/traininglines_"+model_name+'.csv', 'r') as tl:
    csvreader = csv.reader(tl)
    with open(f"{DATA_ROOT}/CSVs/trainingdata_"+model_name+'.csv','w') as td:
      csvwriter = csv.writer(td)
      for row in csvreader:
        rowcopy = row.copy()
        splitsentence = rowcopy[0].split()
        target = rowcopy[1]
        for i, word in enumerate(splitsentence):
          if word == target:
            splitsentence[i] = "[MASK]"
        maskedsentence = sep.join(splitsentence)
        csvwriter.writerow([maskedsentence] + [target])
  return

#a measure of the difficulty of the task- how many training examples and how many training lines
def get_complexity(model_name):
  labeldict = {} #dictionary to count the occurences of the target words
  if model_name == "microsoft/deberta-base":
    model_name = "deberta" #model name is used in naming files, the / character implies a directory which doesn't exist
  with open(f"{DATA_ROOT}/CSVs/trainingdata_"+model_name+'.csv','r') as td:
    csvreader = csv.reader(td)
    i = 0
    for row in csvreader:
      i += 1 #count how many lines are in the file
      target = row[1]
      if target not in labeldict:
        labeldict[target] = 1
      else:
        labeldict[target] += 1

  sorted_labellist = sorted(labeldict, key=labeldict.get, reverse = True)
  sorted_vallist = []
  sorted_labeldict = {}
  for word in sorted_labellist:
    sorted_labeldict[word] = labeldict[word]/i #average fraction of total training examples with this token - more for common words, less for others
    sorted_vallist += [labeldict[word]/i]
  #print(sorted_labeldict)
  #print(sorted_vallist)
  #print(sum(sorted_vallist[:10]))
  return str(len(sorted_labellist)) + " distinct labels, "+ str(i) + " lines; "+str(i/len(sorted_labellist))+" lines per label "

model_names = ["gpt2","bert-base-cased","microsoft/deberta-base"]

for model_name in model_names: #get the tokenizer, save it, generate training data, save that, for every model
  update_tokenizer_get_training_data(model_name, sorted_wordlist, master_keywords)

for model_name in model_names: #get the average number of training examples for each label, for each model
  print(get_complexity(model_name))
