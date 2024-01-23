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



print("Loading Models....")
model_name = "msmarco-distilbert-base-v3"


model_save_path_c1 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_c1", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_c2 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_c2", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_comb = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_comb", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_s1 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_s1", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_s2 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_s2", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_s3 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_s3", "{}-v1-{}".format(model_name, "SCI-FACT"))
model_save_path_seq_v2 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_seq_v2", "{}-v1-{}".format(model_name, "SCI-FACT"))


def eval_train_model(model_path, corpus, test_data, qrels):
  evaluator_model = DRES(models.SentenceBERT(model_path, batch_size=16))
  evaluator = EvaluateRetrieval(evaluator_model, score_function="dot")
  results = evaluator.retrieve(corpus, test_data)
  ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [5,10,25])
  mrr = evaluator.evaluate_custom(qrels, results, [5,10,25], 'mrr@k')
  return ndcg, _map, recall, precision, mrr

print("Evaluating Models....")
c1 = eval_train_model(model_save_path_c1, new_corpus, dict_te_q, new_test_qrels)
print("1 Models done evaluating...")
c2 = eval_train_model(model_save_path_c2, new_corpus, dict_te_q, new_test_qrels)
comb = eval_train_model(model_save_path_comb, new_corpus, dict_te_q, new_test_qrels)
print("Halfway done evaluating....")

s1 = eval_train_model(model_save_path_s1, new_corpus, dict_te_q, new_test_qrels)
s2 = eval_train_model(model_save_path_s2, new_corpus, dict_te_q, new_test_qrels)
s3 = eval_train_model(model_save_path_s3, new_corpus, dict_te_q, new_test_qrels)
seq_v2 = eval_train_model(model_save_path_seq_v2, new_corpus, dict_te_q, new_test_qrels)


print("c1:", c1)
print("c2:", c2)
print("comb:", comb)
print("s1:", s1)
print("s2:", s2)
print("s3:", s3)
print("seq_v2:", seq_v2)