#Import Libraries
import pandas as pd

print("Implementing PyTerrier.....")
#Implement PyTerrier
import pyterrier as pt
from pyterrier.measures import *
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

print("Loading Data....")
# Load Data
#documents
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})


#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test2 = df_test[['qid', 'query']]
df_test2.to_csv('./scifact/BM25_test_queries.csv', sep = '\t', index=False, header=False)
test_query = pt.io.read_topics('./scifact/BM25_test_queries.csv', format='singleline')




#qrels
test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)
test_qrels = test_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
test_qrels['iteration'] = 0
test_qrels = test_qrels.astype({'label': 'int32'})



print("Loading already build index for documents.....")
# Loading already build index
index_ref = pt.IndexRef.of("./indexes/both/data.properties")
index = pt.IndexFactory.of(index_ref)



print("Building Sparse IR Systems....")
#Build Sparse IR Systems
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


print("Evaluating Sparse IR models.....")
# Evaluate models on queries using PyTerrier Experiment Interface

res = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_query, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res.to_csv('results/sparse_ir_res.txt',  sep='\t', index=False)
