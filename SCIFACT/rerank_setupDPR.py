#Import Libraries
import pickle 
import pandas as pd

import pathlib, os
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.beir.retrieval import models


print("Implementing PyTerrier.....")
#Implement PyTerrier
from pyterrier.measures import *
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])




print("Loading Data....")
# Load Data
corpus = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)

#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test = df_test[['qid', 'query']]
test_q = df_test.drop_duplicates()
test_q = test_q.astype({'qid': 'string', 'query' : 'string'})

print(len(test_q))



print("Rewriting to dictionaries....")
#Rewrite dataframes to dict.
new_corpus = {}
for i in range(len(corpus)):
    key = str(corpus['_id'][i])
    new_corpus[key] = {'text' : corpus['text'][i], 'title' : corpus['title'][i]}

def dict_query(data):
    new_dict = {}
    for i in range(len(data)):
      key = str(data['qid'].iloc[i])
      new_dict[key] = data['query'].iloc[i]

    return new_dict


dict_te_q = dict_query(test_q)


print("Loading Models....")
model_name = "msmarco-distilbert-base-v3"

model_save_path_norm = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_norm", "{}-v1-{}".format(model_name, "SCI-FACT"))

def eval_train_model(model_path, corpus, test_data):
  evaluator_model = DRES(models.SentenceBERT(model_path, batch_size=16))
  evaluator = EvaluateRetrieval(evaluator_model, [200], score_function="dot")
  results = evaluator.retrieve(corpus, test_data)
  return results

print("Evaluating Models....")
norm = eval_train_model(model_save_path_norm, new_corpus, dict_te_q)


with open('./scifact/rerank/dpr_results.pkl', 'wb') as f:
    pickle.dump(norm, f)
        
