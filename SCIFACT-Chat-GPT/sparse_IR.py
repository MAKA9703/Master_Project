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



df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test2 = df_test[['qid', 'query']]
df_test2.to_csv('./scifact/bm25_queries.csv', sep = '\t', index=False, header=False)
test_query = pt.io.read_topics('./scifact/bm25_queries.csv', format='singleline')

#test_data 
test_query1 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query1.csv', format='singleline') 
test_query2 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query2.csv', format='singleline')
test_query3 = pt.io.read_topics('./scifact/rephrase/CGPT_test_query3.csv', format='singleline')

test_alt1 = pd.DataFrame()
test_alt1['qid'] = df_test['qid']
test_alt1['query'] = test_query['query'] + " " + test_query['query'] + " " + test_query['query'] + " " + test_query['query'] + " " + test_query['query'] + " " + test_query1['query'] 

test_alt3 = pd.DataFrame()
test_alt3['qid'] = df_test['qid']
test_alt3['query'] = test_query['query'] + " " + test_query1['query'] + " " + test_query['query'] + " " + test_query2['query'] + " " + test_query['query'] + " " + test_query3['query']  



test_comb1 = pd.DataFrame()
test_comb1['qid'] = df_test['qid']
test_comb1['query'] = test_query['query'] + " " + test_query1['query'] 

test_comb2 = pd.DataFrame()
test_comb2['qid'] = df_test['qid']
test_comb2['query'] = test_query['query'] + " " + test_query1['query'] + " " + test_query2['query'] 

test_comb3 = pd.DataFrame()
test_comb3['qid'] = df_test['qid']
test_comb3['query'] = test_query['query'] + " " + test_query1['query'] + " " + test_query2['query'] + " " + test_query3['query'] 


#qrels
test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)
test_qrels = test_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
test_qrels['iteration'] = 0



print("Loading already build index for documents.....")
# Loading already build index
index_ref = pt.IndexRef.of("./indexes/both/data.properties")
index = pt.IndexFactory.of(index_ref)



print("Building Sparse IR Systems....")
#Build Sparse IR Systems
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


print("Evaluating Sparse IR models.....")
# Evaluate models on queries using PyTerrier Experiment Interface
test_qrels = test_qrels.astype({'label': 'int32'})



res_alt1 = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_alt1, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res_alt1.to_csv('results/sparse_ir_res_alt1.txt',  sep='\t', index=False)

print("1 out of 5 done!")
res_alt3 = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_alt3, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res_alt3.to_csv('results/sparse_ir_res_alt3.txt',  sep='\t', index=False)
print("2 out of 5 done!")

res_comb1 = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_comb1, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res_comb1.to_csv('results/sparse_ir_res_comb1.txt',  sep='\t', index=False)
print("3 out of 5 done!")
res_comb2 = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_comb2, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res_comb2.to_csv('results/sparse_ir_res_comb2.txt',  sep='\t', index=False)
print("4 out of 5 done!")
res_comb3 = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_comb3, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_25", "map_cut_5", "map_cut_10", "map_cut_25", "P_5", "P_10", "P_25", "recall_5", "recall_10", "recall_25", MRR@10])

res_comb3.to_csv('results/sparse_ir_res_comb3.txt',  sep='\t', index=False)
print("5 out of 5 done!")