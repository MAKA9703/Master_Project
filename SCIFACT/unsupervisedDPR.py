#Import libraries
import pandas as pd
import json

from beir.beir.retrieval import models
from beir.beir.retrieval.evaluation import EvaluateRetrieval
from beir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

print("Implementing PyTerrier.....")
#Implement PyTerrier
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])




corpus = pd.read_json('./scifact/corpus.jsonl', lines=True)


df_test = pd.read_csv('./scifact/test.csv', sep='\t')
test_query = df_test[['qid', 'query']]
test_query = test_query.drop_duplicates()

df_test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t')



#Rewrite pandas data-frame into correctly formatted dictionary
#Note that _id here refers to doc id.
new_corpus = {}
for i in range(len(corpus)):
    key = str(corpus['_id'][i])
    new_corpus[key] = {'text' : corpus['text'][i], 'title' : corpus['title'][i]}

def trans_test_queries(df):
    new_test_queries = {}
    for i in range(len(df)):
        key = str(df['qid'].iloc[i])
        new_test_queries[key] = df['query'].iloc[i]
    return new_test_queries

dict_te_q = trans_test_queries(test_query)


#test_qrels
new_test_qrels = {}
for i in range(len(df_test_qrels)):
    query_key = str(df_test_qrels['query-id'][i])
    new_test_qrels[query_key] = {str(df_test_qrels['corpus-id'][i]) : int(df_test_qrels['score'][i])}


def retrieve_results(corpus, df_dict, qrels, name):
    model_dpr = DRES(models.SentenceBERT("msmarco-distilbert-base-v3", batch_size=16))
    retriever_dpr = EvaluateRetrieval(model_dpr, score_function="dot")
    results_dpr = retriever_dpr.retrieve(corpus, df_dict)
    ndcg, _map, recall, precision = retriever_dpr.evaluate(qrels, results_dpr, [5,10,25])
    mrr = retriever_dpr.evaluate_custom(qrels, results_dpr, [10], 'mrr@k')

    with open(f'./results/unsupervisedDPR_{name}.txt', 'w') as f: 
     f.write(json.dumps(ndcg))
     f.write('\n')
     f.write(json.dumps(_map))
     f.write('\n')
     f.write(json.dumps(precision))
     f.write('\n')
     f.write(json.dumps(recall))
     f.write('\n')
     f.write(json.dumps(mrr))
     f.close()

retrieve_results(new_corpus, dict_te_q, new_test_qrels, "res")