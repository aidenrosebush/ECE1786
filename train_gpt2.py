"""Fine-tune gpt2 to predict redacted tokens.

Part of RedactedGPT. Originally developed as Colab notebooks; converted to scripts.
Paths are resolved from DATA_ROOT in config.py — set REDACTEDGPT_DATA to override.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, MODEL_ROOT  # noqa: E402

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
import string
import numpy as np
import torch


# Commented out IPython magic to ensure Python compatibility.

# Read CSV as Pandas DataFrame
data_df = pd.read_csv("trainingdata_gpt2.csv", header=None, names=["sentence", "label"])

# Remove trailing punctuation from masked token labels

def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

data_df['label'] = data_df['label'].apply(remove_punctuation)

data_df

model_name = "gpt2" # Just change this to try out different models
tokenizer = AutoTokenizer.from_pretrained(f"{DATA_ROOT}/Tokenizer Files/gpt2_tokenizer") #load just the tokenizer



# Add masked token labels to vocabulary (cannot be split into multiple tokens)
# only needs to be run when trying out a new model or training set

# for i, word in enumerate(data_df["label"].values):
#   if word not in tokenizer.vocab:
#     tokenizer.add_tokens(word)

#tokenizer.save_pretrained(f"{DATA_ROOT}/Tokenizer Files/gpt2_tokenizer") #save the tokenizer

# tokenizer = AutoTokenizer.from_pretrained("drive/MyDrive/Project/Tokenizer Files/gpt2_tokenizer")

# data_df = data_df[:20] # For testing purposes

# Convert Pandas DataFrame to HuggingFace Dataset
full_dataset = Dataset.from_pandas(data_df)
full_dataset = full_dataset.shuffle(seed=42)
# Split dataset into train-test-validation sets
train_testvalid = full_dataset.train_test_split(test_size =0.3)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size = 0.65)
# gather everyone if you want to have a single DatasetDict
data_ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})
data_ds

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

tokenized_data_ds = data_ds.map(tokenize_function, batched= True)

tokenized_data_ds["train"]["label"][0] # All -100 except 1

eval_len = tokenized_data_ds['val'].num_rows

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  # Get ground truths of masked tokens and predicted tokens
  masked_tokens_gts = []
  masked_tokens_inds = []
  for label in labels:
    for i in range(len(label)):
      if label[i] != -100:
        masked_tokens_inds += [i]
        masked_tokens_gts += [label[i]]*10 #multiply by k = 10 to accomodate top k predictions
        break
  # Get predicted masked token (index = token_id)
  predicted_sent = torch.topk(torch.tensor(logits), 10, dim=2).indices #do topk with k = 10
  predicted_masked_tokens = []
  for i, ind in enumerate(masked_tokens_inds):
    predicted_masked_tokens += predicted_sent[i][ind] #get whole flattened list of topk predictions
  # Compute accuracy
  accuracy = 10*sum(masked_tokens_gts == np.array(predicted_masked_tokens))/len(masked_tokens_gts) #multipy by k to compensate for flattening
  return {'accuracy': accuracy}

# Initialize model and training arguments

initial_model = AutoModelForCausalLM.from_pretrained(model_name)
initial_model.config.vocab_size = tokenizer.vocab_size
initial_model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = model.config.eos_token_id

training_args = TrainingArguments(
    output_dir=f"{DATA_ROOT}/Models/GPT",
    overwrite_output_dir=True,
    evaluation_strategy = "steps",
    per_device_eval_batch_size= 100,
    eval_steps = 1/20, # Can just be a ratio less than 1 of the total training steps
    logging_steps = 1/20,
    save_total_limit=1,
    save_strategy="epoch",
    per_device_train_batch_size = 32,
    learning_rate = 5e-4,
    eval_accumulation_steps = 2 #this is the parameter that prevents GPU memory overflow (unload eval results to CPU memory)
)
class EvalCallback(TrainerCallback):
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

# trainer.save_model(f"{DATA_ROOT}/Models/GPT")

#test a saved model from model_dir, with training arguments from training set to RAM doesn't overflow
#with known maxmimum val batch size to prevent RAM overflow
def test(model_dir, training_args, tokenized_data_ds, val_size):
  model = AutoModelForCausalLM.from_pretrained(model_dir)
  model.eval()

  tester = Trainer(
    model = model,
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
    if n > val_size:
      total_samples += val_size
      test_sets = test_set.train_test_split(test_size = val_size/n)
      n = test_sets['train'].num_rows
      accuracy += val_size*tester.evaluate(test_sets['test'])['eval_accuracy']
      test_set = test_sets['train']
    else:
      total_samples += test_set.num_rows
      accuracy += test_set.num_rows*tester.evaluate(test_set)['eval_accuracy']
      n = 0
  return "accuracy: "+str(accuracy/total_samples)
# print(test(f"{DATA_ROOT}/Models/GPT", training_args, tokenized_data_ds, 1674))

def get_samples(model_dir, tokenized_data_ds):
  model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_dir)
  model.eval()

  for i in range(100):
    ids = torch.tensor([tokenized_data_ds['test']['input_ids'][i]])
    label_ids = tokenized_data_ds['test']['label'][i]
    mask_id = 0
    for j, id in enumerate(label_ids):
      if id != -100:
        label = tokenizer.convert_ids_to_tokens(id)
        mask_id = j
    trimmed_ids_nonzero_idx = ids[0].clone().detach().nonzero() #get rid of pad tokens for demonstration purposes
    trimmed_ids = ids[0].clone().detach()[trimmed_ids_nonzero_idx]
    input_sentence = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(trimmed_ids))
    output = model(ids)
    logits = output.logits[0][mask_id]
    probs, ids = torch.sort(torch.softmax(logits, dim = 0), dim = 0, descending = True)

    print("Input Sentence: "+ input_sentence)
    print("Correct Label: "+label)
    print("Probabilities: "+str(probs[:10].detach().numpy()))
    print("Top 10 Guesses: "+str(tokenizer.convert_ids_to_tokens(ids[:10])))
    print("\n")
  return

get_samples(f"{DATA_ROOT}/Models/GPT", tokenized_data_ds)
