import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


#train data
df_train = pd.read_csv('./scifact/train.csv', sep='\t', dtype=str)
df_train2 = df_train[['qid', 'query']]
df_train2.to_csv('./scifact/temp_train_queries.csv', sep = '\t', index=False, header=False)
train_query = pt.io.read_topics('./scifact/temp_train_queries.csv', format='singleline')
train_source = train_query['query']


#INSPIRED BY!
#https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(15)

#Load pre-trained model.
#"Vamsi/T5_Paraphrase_Paws"
model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

#Use cuda if possible.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)



#Use model to paraphrase each of the queries in the training data-set.
all_train_outputs = []
for i in range(len(train_source)):
  text =  "paraphrase: " + train_source.iloc[i] + " </s>"

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=256,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=3
  )

  final_outputs =[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != train_source.iloc[i].lower() and sent not in final_outputs:
          final_outputs.append(sent)
  all_train_outputs.append(final_outputs)

  if i == 0:
    print ("\nOriginal Question ::")
    print (train_source.iloc[i])
    print ("\n")
    print ("Paraphrased Questions :: ")
    for i, final_output in enumerate(final_outputs):
        print("{}: {}".format(i, final_output))

  if i % 50 == 0:
    print(i,  " out of ", len(train_source))


#Get a list for each of the 5 different ways each query has been rewritten.
#If a query was not able to be rewritten 5 different ways, then fill remaining slots with index 0 query.
train_rewritten_text1 = []
train_rewritten_text2 = []
train_rewritten_text3 = []


for i in range(len(all_train_outputs)):
  train_rewritten_text1.append(all_train_outputs[i][0])

for i in range(len(all_train_outputs)):
  if len(all_train_outputs[i]) < 2:
    train_rewritten_text2.append(all_train_outputs[i][0])
  else:
    train_rewritten_text2.append(all_train_outputs[i][1])

for i in range(len(all_train_outputs)):
  if len(all_train_outputs[i]) < 3:
    train_rewritten_text3.append(all_train_outputs[i][0])
  else:
    train_rewritten_text3.append(all_train_outputs[i][2])



# Create dataframes for rewritten queries.
train_new_queries = pd.DataFrame()
train_new_queries['qid'] = train_query['qid']
train_new_queries['query'] = train_source

train_new_queries1 = pd.DataFrame()
train_new_queries1['qid'] = train_query['qid']
train_new_queries1['query'] = train_rewritten_text1

train_new_queries2 = pd.DataFrame()
train_new_queries2['qid'] = train_query['qid']
train_new_queries2['query'] = train_rewritten_text2

train_new_queries3 = pd.DataFrame()
train_new_queries3['qid'] = train_query['qid']
train_new_queries3['query'] = train_rewritten_text3



#Replace function to remove special characters from queries.
def replace_func(df):
    df['query'] = df['query'].str.replace('\t',' ')
    df['query'] = df['query'].str.replace('\n',' ')
    return df 

#Clean queries for special characters that some models might not be able to handle.
train_new_queries = replace_func(train_new_queries)
train_new_queries1 = replace_func(train_new_queries1)
train_new_queries2 = replace_func(train_new_queries2)
train_new_queries3 = replace_func(train_new_queries3)

#Save files
train_new_queries1.to_csv('./scifact/rephrase/T5_train_query1.csv', sep='\t', index = False, header = False)
train_new_queries2.to_csv('./scifact/rephrase/T5_train_query2.csv', sep='\t', index = False, header = False)
train_new_queries3.to_csv('./scifact/rephrase/T5_train_query3.csv', sep='\t', index = False, header = False)

