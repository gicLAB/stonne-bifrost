import transformers
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from modeling_bert import BertModel
import torch
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
class_names = ['negative', 'neutral', 'positive']
MAX_LEN = 160


# This function uses BERT to create a sentiment classifier that we will use in this benchmark
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes, simulation_file, tiles_path, sparsity_ratio, stats_path):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, sim_file=simulation_file, tiles=tiles_path, sparsity=sparsity_ratio, stats_path=stats_path)
    #self.bert.set_simulator_params(simulation_file, tiles_path, sparsity_ratio, stats_path)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


def run_model(simulation_file='../../../simulation_files/sigma_128mses_64_bw.cfg', tiles_path='tiles/128_mses/', sparsity_ratio=0.0, stats_path='', trained_weights='best_model_state.bin'):
    # Creating the instances and running the model 
    model = SentimentClassifier(len(class_names), simulation_file, tiles_path, sparsity_ratio, stats_path)
    print(model)
    model_dict = model.state_dict()
    state_dict = torch.load(trained_weights, map_location=torch.device('cpu'))
    # Due to some changes in BERT version state dict misses a Sequential vector. Let's include it in the pretrained weights to make sure the weights are loading correctly
    from collections import OrderedDict
    pretrained_dict=OrderedDict()
    for k, v in model_dict.items():
        if k in state_dict:
            pretrained_dict[k]=state_dict[k]
        else:
            pretrained_dict[k]=model_dict[k]
    #print(model.bert.embeddings)
    #print(pretrained_dict)

    #print(state_dict)
    model.load_state_dict(pretrained_dict)
    #print(model)

    # Running the model
    review_text = "The movie was actually very good!!!"

    # We have to use the tokenizer to encode the text:

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    encoded_review = tokenizer.encode_plus(
      review_text,
      max_length=MAX_LEN,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )

    # Let's get the predictions from our model
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print(output)
    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')

# main
#run_model()
