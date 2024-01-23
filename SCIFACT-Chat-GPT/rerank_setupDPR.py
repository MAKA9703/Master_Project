#Import Libraries
import pickle 
import pandas as pd

import pathlib, os
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.beir.retrieval import models


import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

print("Loading Data....")
# Load Data
corpus = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)

#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test = df_test[['qid', 'query']]
test_query = df_test.drop_duplicates()


#test_data 
test_query1 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query1.csv', format='singleline')
test_query1 = test_query1.drop_duplicates(subset=['qid'])
test_query1 = test_query1.astype({'qid': 'string', 'query':'string'})

test_query2 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query2.csv', format='singleline')
test_query2 = test_query2.drop_duplicates(subset=['qid'])
test_query2 = test_query2.astype({'qid': 'string', 'query':'string'})

test_query3 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query3.csv', format='singleline')
test_query3 = test_query3.drop_duplicates(subset=['qid'])
test_query3 = test_query3.astype({'qid': 'string', 'query':'string'})


test_alt3 = pd.DataFrame()
test_alt3['qid'] = test_query['qid']
test_alt3['query'] = test_query['query'] + " " + test_query1['query'] + " " + test_query['query'] + " " + test_query2['query'] + " " + test_query['query'] + " " + test_query3['query']  


test_comb3 = pd.DataFrame()
test_comb3['qid'] = test_query['qid']
test_comb3['query'] = test_query['query'] + " " + test_query1['query'] + " " + test_query2['query'] + " " + test_query3['query'] 



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


dict_te_q = dict_query(test_query)
dict_te_q_alt3 = dict_query(test_alt3)
dict_te_q_comb3 = dict_query(test_comb3)


print("Loading Models....")
def eval_train_model(model_path, corpus, test_data):
  evaluator_model = DRES(models.SentenceBERT(model_path, batch_size=16))
  evaluator = EvaluateRetrieval(evaluator_model, [200], score_function="dot")
  results = evaluator.retrieve(corpus, test_data)
  return results



def eval_res(name, corpus, dict_test, store_name):
  model_name = "msmarco-distilbert-base-v3"
  model_save_path = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), f"dpr_models/output_{name}", "{}-v1-{}".format(model_name, "SCI-FACT"))
  res = eval_train_model(model_save_path, corpus, dict_test)
  with open(f'./scifact/rerank/dpr_results_{store_name}.pkl', 'wb') as f:
    pickle.dump(res, f)
  f.close()

print("Evaluating Models....")
eval_res("c1", new_corpus, dict_te_q, 'c1')
eval_res("c1", new_corpus, dict_te_q_alt3, 'c1_alt3')
eval_res("c1", new_corpus, dict_te_q_comb3, 'c1_comb3')

eval_res("c2", new_corpus, dict_te_q, 'c2')
eval_res("c2", new_corpus, dict_te_q_alt3, 'c2_alt3')
eval_res("c2", new_corpus, dict_te_q_comb3, 'c2_comb3')

eval_res("seq_v2", new_corpus, dict_te_q, 'seq_v2')
eval_res("seq_v2", new_corpus, dict_te_q_alt3, 'seq_v2_alt3')
eval_res("seq_v2", new_corpus, dict_te_q_comb3, 'seq_v2_comb3')




        
