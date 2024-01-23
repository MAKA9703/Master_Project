import pandas as pd 

import pathlib, os
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.beir.retrieval import models

print("Loading Data....")
#LOAD DATA
corpus = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)


#test_data
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_q = df_test[['qid', 'query']]
test_q = test_q.drop_duplicates()

#test qrels
df_test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)



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

#test_qrels
new_test_qrels = {}
for i in range(len(df_test_qrels)):
    query_key = str(df_test_qrels['query-id'][i])
    new_test_qrels[query_key] = {str(df_test_qrels['corpus-id'][i]) : int(df_test_qrels['score'][i])}



def eval_trained_model(model_path, corpus, test_data, qrels):
  evaluator_model = DRES(models.SentenceBERT(model_path, batch_size=16))
  evaluator = EvaluateRetrieval(evaluator_model, score_function="dot")
  results = evaluator.retrieve(corpus, test_data)
  ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [5,10,25])
  mrr = evaluator.evaluate_custom(qrels, results, [5,10,25], 'mrr@k')
  return ndcg, _map, recall, precision, mrr



def eval_res(name, corpus, dict_test, qrels):
  model_name = "msmarco-distilbert-base-v3"
  model_save_path = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), f"dpr_models/output_{name}", "{}-v1-{}".format(model_name, "SCI-FACT"))
  res = eval_trained_model(model_save_path, corpus, dict_test, qrels)
  return res 

c1 = eval_res("c1", new_corpus, dict_te_q, new_test_qrels)
c2 = eval_res("c2", new_corpus, dict_te_q, new_test_qrels)
comb = eval_res("comb", new_corpus, dict_te_q, new_test_qrels)

s1 = eval_res("s1", new_corpus, dict_te_q, new_test_qrels)
s2 = eval_res("s2", new_corpus, dict_te_q, new_test_qrels)
s3 = eval_res("s3", new_corpus, dict_te_q, new_test_qrels)

seq_v2 = eval_res("seq_v2", new_corpus, dict_te_q, new_test_qrels)
print("c1:", c1)
print("c2:", c2)
print("comb:", comb)


print("s1:", s1)
print("s2:", s2)
print("s3:", s3)
print("seq_v2:", seq_v2)