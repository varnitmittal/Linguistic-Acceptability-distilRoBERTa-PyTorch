#########################
from flask import Flask, request, render_template, jsonify, send_file
import os

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
#########################

#########################
#regular expressions
import re 
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])" ###

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text) ###
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
#########################

#!pip install transformers
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import numpy as np

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

"""# Drivers"""

MAX_LEN = 64
directory = './my_model_save/'

"""# Input Preprocessing"""

def getTokenizer(directory):
  tokenizer = RobertaTokenizer.from_pretrained(directory)
  return tokenizer

def queryEncoder(sentences, tokenizer): 
  input_ids = []
  for each in sentences:
    encoded_each = tokenizer.encode(
        text = each,
        add_special_tokens = True
    )
    input_ids.append(np.array(encoded_each))
  return input_ids

def encodedQueryPadding(input_ids, MAX_LEN):
  padded_input_ids = []
  for index in range(input_ids.shape[0]):
      padded = np.zeros((MAX_LEN,), dtype=np.int64)
      if len(input_ids[index]) < MAX_LEN:
        padded[:len(input_ids[index])] = input_ids[index][:]
        padded_input_ids.append(padded)
      else: 
        padded_input_ids.append(input_ids[index][:MAX_LEN])
  return padded_input_ids

def tokenAttentionMasking(padded_input_ids):
  attention_masks = []
  for index in range(padded_input_ids.shape[0]):
    att_mask = [int(each > 0) for each in padded_input_ids[index]]
    attention_masks.append(att_mask)
  return attention_masks

def inputPreprocessing(query, directory, MAX_LEN):
  tokenizer = getTokenizer(directory)
  input_ids = np.array(queryEncoder(query, tokenizer))
  padded_input_ids = np.array(encodedQueryPadding(input_ids, MAX_LEN))
  attention_masks = np.array(tokenAttentionMasking(padded_input_ids))
  
  return padded_input_ids, attention_masks

"""# Getting Model"""

def getModel(directory, device):
  model = RobertaForSequenceClassification.from_pretrained(directory)
  
  model.to(device)
  for param in model.parameters():
    param.requires_grad = False
  model.eval()
  
  print("model sent to:",device)  
  return model

#model = getModel(directory, device)

"""# Get Predictions"""

def convertToTensor(obj):
  return torch.tensor(obj)

def getPredictions(model):
  padded_input_ids, attention_masks = inputPreprocessing(query=query, 
                                                       directory=directory, 
                                                       MAX_LEN=MAX_LEN)
  padded_input_ids = convertToTensor(padded_input_ids)
  attention_masks = convertToTensor(attention_masks)
  
  with torch.no_grad():
    pred = model(padded_input_ids.to(device), 
                 attention_mask=attention_masks.to(device), 
                 labels=None)

  logits = pred[0].detach().cpu().numpy()
  return logits

def activateLogits(model):
  logit_predictions = getPredictions(model)
  sig_logit = np.array(torch.sigmoid(torch.tensor(logit_predictions)))
  sig_logit_arg = np.argmax(sig_logit, axis=1).flatten()
  sig_logit_arg_value = np.max(sig_logit, axis=1).flatten()
  return sig_logit_arg, sig_logit_arg_value


#########################
@app.route("/", methods=['GET'])
def index():
    return render_template("home.html")
#########################

#########################
@app.route("/predict", methods=["POST"])
def predict():
  json_data = request.get_json()
  global query
  if json_data["received"]:
    #print(json_data["raw_text"])
    query = split_into_sentences(json_data["raw_text"])
    final_predictions, final_predictions_confidence = activateLogits(model)

    prediction_dictionary = {}
    for i in range(len(final_predictions)):
      prediction_dictionary[i] = [int(final_predictions[i]),
                                  query[i],
                                  float(final_predictions_confidence[i])]

    response = jsonify({
      'success': True,
      'model_output': prediction_dictionary
    })
    response.status_code = 200
    #print("Done!")
    return response
    #return jsonify({"Done": True})
  else:
    response = jsonify({
      'success': False 
    })
    #print("Done!")
    return response
#########################


global model
model = getModel(directory, device)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(port)
