"""Fine-tune bert-base-cased to predict redacted tokens.

Part of RedactedGPT. Originally developed as Colab notebooks; converted to scripts.
Paths are resolved from DATA_ROOT in config.py — set REDACTEDGPT_DATA to override.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, MODEL_ROOT  # noqa: E402

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, TrainerCallback, PreTrainedModel
import string
import numpy as np
import torch


# Commented out IPython magic to ensure Python compatibility.

# Read CSV as Pandas DataFrame
data_df = pd.read_csv("trainingdata_bert-base-cased.csv", header=None, names=["sentence", "label"])

# Remove trailing punctuation from masked token labels

def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

data_df['label'] = data_df['label'].apply(remove_punctuation)

data_df

model_name = "bert-base-cased" # Just change this to try out different models
tokenizer = AutoTokenizer.from_pretrained(f"{DATA_ROOT}/Tokenizer Files/BERT_tokenizer") #load just the tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add mask token

print(tokenizer.tokenize("MEMORANDUM FOR: Inspector General, CIA")) #test the tokenizer- notice CIA is a keyword and has its own token

# data_df = data_df[:20] # For testing purposes

# Convert Pandas DataFrame to HuggingFace Dataset
full_dataset = Dataset.from_pandas(data_df)
full_dataset = full_dataset.shuffle(seed=42)
# Split dataset into train-test-validation sets
train_testvalid = full_dataset.train_test_split(test_size =0.3)
# Split the 10% test + valid in half test, half valid. Different ratio for other models depending on memory requirement. 70-15-15 split overall here.
test_valid = train_testvalid['test'].train_test_split(test_size = 0.5)
# gather everyone if you want to have a single DatasetDict
data_ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
data_ds['train']['sentence'][0]

# Apply tokenizer
def tokenize_function(batch):
  # Tokenize sentences
  encoded_batch = tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)
  # Create label for each sentence: -100 for unmasked token, actual ID for masked token (list)
  tokenized_sentences = encoded_batch.input_ids.copy()
  encoded_labels = []
  for i, sent in enumerate(tokenized_sentences):
    encoded_labels += [[-100]*len(sent)]
    for j, token_id in enumerate(sent):
      if token_id == tokenizer.mask_token_id:
        if tokenizer.tokenize(batch["label"][i]) == []:
          label = " "
        else:
          label = tokenizer.tokenize(batch["label"][i])[0]
        encoded_labels[-1][j] = tokenizer.convert_tokens_to_ids(label)
        break # Can break as only one mask per sentence
  encoded_batch["label"] = encoded_labels
  return encoded_batch

tokenized_data_ds = data_ds.map(tokenize_function, batched=True)

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  # Get ground truths of masked tokens and predicted tokens
  masked_tokens_gts = []
  masked_tokens_inds = []
  for label in labels:
    for i in range(len(label)):
      if label[i] != -100:
        masked_tokens_inds += [i]
        masked_tokens_gts += [label[i]]*10 #use topk with k = 10, so if any of these 10 correct labels match with a correct guess, consider it a hit
        break
  # Get predicted masked token (index = token_id)
  predicted_sent = torch.topk(torch.tensor(logits), 10, dim=2).indices #implement topk
  predicted_masked_tokens = []
  for i, ind in enumerate(masked_tokens_inds): #i: number of the example in the batch; ind: index within each example of the masked token
    predicted_masked_tokens += predicted_sent[i][ind]
  # Compute accuracy
  accuracy = 10*sum(masked_tokens_gts == np.array(predicted_masked_tokens))/len(masked_tokens_gts) #multiply accuracy by ten to compensate for extra labels
  return {'accuracy': accuracy}

#Initialize model and training arguments

initial_model = AutoModelForMaskedLM.from_pretrained(model_name)
initial_model.config.vocab_size = tokenizer.vocab_size
initial_model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = model.config.eos_token_id

training_args = TrainingArguments(
    output_dir=f"{DATA_ROOT}/Models/BERT",
    overwrite_output_dir=True,
    evaluation_strategy = "steps",
    per_device_eval_batch_size= 50, #fine tuned: larger = faster but uses more RAM. Must run with T4 high RAM
    eval_steps = 1/9,
    logging_steps = 1/9,
    save_strategy="no",
    num_train_epochs = 3,
    per_device_train_batch_size = 50, #fine tuned
    gradient_accumulation_steps = 6, #prevents GPU RAM overflow during training; pass gradient to CPU
    eval_accumulation_steps = 1, #unloads val results to CPU memory after this many steps. larger = faster but uses more RAM
    learning_rate = 5e-5
)

# Initialize trainer
class EvalCallback(TrainerCallback): #callback to prevent the model from accumulating gradients when doing eval. this prevents RAM overflow
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
      kwargs['model'].eval()
      pass
    def on_log(self, args, state, control, logs=None, **kwargs):
      kwargs['model'].train()
      pass
ee = EvalCallback()
trainer = Trainer(
    model = initial_model,
    args = training_args,
    train_dataset=tokenized_data_ds["train"],
    eval_dataset=tokenized_data_ds["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [ee]
)
# Train model
# trainer.train()
#trainer.save_model("drive/MyDrive/Project/Models/Bert")

# trainer.save_model(f"{DATA_ROOT}/Models/Bert")

#test a saved model from model_dir, with training arguments from training set to RAM doesn't overflow
#with known maxmimum val batch size to prevent RAM overflow
def test(model_dir, training_args, tokenized_data_ds, val_size):
  model = AutoModelForMaskedLM.from_pretrained(model_dir)
  model.eval()

  tester = Trainer(
    model = initial_model,
    args = training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
  )
  accuracy = 0.0
  total_samples = 0
  test_set = tokenized_data_ds['test']
  n = test_set.num_rows
  while n > 0:
    print(n)
    if n > val_size: #too many samples for evaluation; would cause memory overflow
      total_samples += val_size
      test_sets = test_set.train_test_split(test_size = val_size/n) #use train test split to ensure too many examples aren't sent to the tester.evaluate function
      n = test_sets['train'].num_rows
      accuracy += val_size*tester.evaluate(test_sets['test'])['eval_accuracy']
      test_set = test_sets['train']
    else: #last time: entire remaining test set can be sent to the evaluate function in one shot
      total_samples += test_set.num_rows
      accuracy += test_set.num_rows*tester.evaluate(test_set)['eval_accuracy']
      n = 0
  return "accuracy: "+str(accuracy/total_samples)
# print(test(f"{DATA_ROOT}/Models/Bert", training_args, tokenized_data_ds, 2000))

#load the model and get the first 100 test samples for qualitative assessment
def get_samples(model_dir, tokenized_data_ds):
  model = AutoModelForMaskedLM.from_pretrained(model_dir)
  model.eval()

  for i in range(100):
    ids = torch.tensor([tokenized_data_ds['test']['input_ids'][i]])
    masks = torch.tensor([tokenized_data_ds['test']['attention_mask'][i]])
    label_ids = tokenized_data_ds['test']['label'][i]
    mask_id = 0
    for j, id in enumerate(label_ids):
      if id != -100:
        label = tokenizer.convert_ids_to_tokens(id)
        mask_id = j
    #get the input sentence
    trimmed_ids_nonzero_idx = ids[0].clone().detach().nonzero() #get rid of pad tokens for demonstration purposes
    trimmed_ids = ids[0].clone().detach()[trimmed_ids_nonzero_idx]
    input_sentence = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(trimmed_ids))

    #get the output probabilites and corresponding words
    output = model(ids, masks)
    logits = output.logits[0][mask_id]
    probs, ids = torch.sort(torch.softmax(logits, dim = 0), dim = 0, descending = True)

    print("Input Sentence: "+ input_sentence)
    print("Correct Label: "+label)
    print("Probabilities: "+str(probs[:10].detach().numpy()))
    print("Top 10 Guesses: "+str(tokenizer.convert_ids_to_tokens(ids[:10])))
    print("\n")
  return

get_samples(f"{DATA_ROOT}/Models/Bert", tokenized_data_ds)

tokenized_data_ds['test']['input_ids'][0]

#experimental interactive function for generating guesses about actual redacted infromation
#not relavant to core conclusions of project; demonstrates infeasiblity of predicting classified info as results never seem plausible
#code like this would be used in a future iteration of the project granted more time, better data processing and more computing resources
def guess(model_dir, sentence):
  model = AutoModelForMaskedLM.from_pretrained(model_dir)
  model.eval()
  s = sentence
  idx = 0
  print(s)

  for i in range(10):
    print(s.split())
    mask_index = s.split().index('[MASK]')
    encoded_sentence = tokenizer(sentence, padding="max_length", truncation=True, max_length=64)
    output = model(torch.tensor([encoded_sentence['input_ids']]), torch.tensor([encoded_sentence['attention_mask']]))
    logits = output.logits[0][-1]
    probs, ids = torch.sort(torch.softmax(logits, dim = 0), dim = 0, descending = True)

    print("Sentence so far: "+s)
    print("Probabilities: "+str(probs[:10].detach().numpy()))
    print("Top 10 Guesses: "+str(tokenizer.convert_ids_to_tokens(ids[:10])))
    print("Enter an index to select a word and generate the next. Enter -1 to produce more guesses.")
    idx = int(input())
    i = 0
    while idx == -1:
      print("Probabilities: "+str(probs[10*(i+1):10*(i+2)].detach().numpy()))
      print("Next 10 Guesses: "+str(tokenizer.convert_ids_to_tokens(ids[10*(i+1):10*(i+2)])))
      print("Enter an index to select a word and generate the next. Enter -1 to produce more guesses.")
      idx = int(input())
      i += 1
    s = s.split()
    s[mask_index] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([ids[idx]]))
    s += [" "]
    for i in range(len(s)-2, mask_index, -1):
      s[i+1] = s[i]
    s[mask_index + 1] = '[MASK]'
    s = ' '.join(s)

guess(f"{DATA_ROOT}/Models/Bert",s2)

s1 = "But even smaller groups, such as the [MASK] posed threats to US interests."
s2 = "Renditions Branch helped capture and render [MASK] "
