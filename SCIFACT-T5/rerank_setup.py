#Import Libraries
import pandas as pd


print("Implementing PyTerrier.....")
#Implement PyTerrier
from pyterrier.measures import *
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])




print("Loading Data....")
# Load Data
#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test2 = df_test[['qid', 'query']]
df_test2.to_csv('./scifact/BM25_test_queries.csv', sep = '\t', index=False, header=False)
test_query = pt.io.read_topics('./scifact/BM25_test_queries.csv', format='singleline')
test_query = test_query.drop_duplicates()



#test_data 
test_query1 = pt.io.read_topics('./scifact/rephrase/T5_test_query1.csv', format='singleline')
test_query1 = test_query1.drop_duplicates(subset=['qid'])
test_query1 = test_query1.astype({'qid': 'string', 'query':'string'})

test_query2 = pt.io.read_topics('./scifact/rephrase/T5_test_query2.csv', format='singleline')
test_query2 = test_query2.drop_duplicates(subset=['qid'])
test_query2 = test_query2.astype({'qid': 'string', 'query':'string'})

test_query3 = pt.io.read_topics('./scifact/rephrase/T5_test_query3.csv', format='singleline')
test_query3 = test_query3.drop_duplicates(subset=['qid'])
test_query3 = test_query3.astype({'qid': 'string', 'query':'string'})

test_alt1 = pd.DataFrame()
test_alt1['qid'] = test_query['qid']
test_alt1['query'] = test_query['query'] + ' ' + test_query['query'] + ' ' + test_query['query'] + ' ' + test_query['query'] + ' ' + test_query['query'] + ' ' + test_query1['query'] 
test_alt1 = test_alt1.astype({'qid': 'string', 'query':'string'})

test_alt3 = pd.DataFrame()
test_alt3['qid'] = test_query['qid']
test_alt3['query'] = test_query['query'] + ' ' + test_query1['query'] + ' ' + test_query['query'] + ' ' + test_query2['query'] + ' ' + test_query['query'] + ' ' + test_query3['query']  
test_alt3 = test_alt3.astype({'qid': 'string', 'query':'string'})

test_comb1 = pd.DataFrame()
test_comb1['qid'] = test_query['qid']
test_comb1['query'] = test_query['query'] + ' ' + test_query1['query'] 
test_comb1 = test_comb1.astype({'qid': 'string', 'query':'string'})

test_comb2 = pd.DataFrame()
test_comb2['qid'] = test_query['qid']
test_comb2['query'] = test_query['query'] + ' ' + test_query1['query'] + ' ' + test_query2['query'] 
test_comb2 = test_comb2.astype({'qid': 'string', 'query':'string'})

test_comb3 = pd.DataFrame()
test_comb3['qid'] = test_query['qid']
test_comb3['query'] = test_query['query'] + ' ' + test_query1['query'] + ' ' + test_query2['query'] + ' ' + test_query3['query'] 
test_comb3 = test_comb3.astype({'qid': 'string', 'query':'string'})

print("Loading already build index for documents.....")
# Loading already build index
index_ref = pt.IndexRef.of("./indexes/both/data.properties")
index = pt.IndexFactory.of(index_ref)



print("Building Sparse IR Systems....")
#Build Sparse IR Systems
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


print("Retrieving results using BM25....")

topk_bm25 = bm25 % 200


retrieved_res = topk_bm25.transform(test_query)

retrieved_res_alt1 = topk_bm25.transform(test_alt1)
retrieved_res_alt3 = topk_bm25.transform(test_alt3)

retrieved_res_comb1 = topk_bm25.transform(test_comb1)
retrieved_res_comb2 = topk_bm25.transform(test_comb2)
retrieved_res_comb3 = topk_bm25.transform(test_comb3)



retrieved_res.to_csv('./scifact/rerank/BM25_res.csv', sep = '\t', index=False, header=True)
retrieved_res_alt1.to_csv('./scifact/rerank/BM25_res_alt1.csv', sep = '\t', index=False, header=True)
retrieved_res_alt3.to_csv('./scifact/rerank/BM25_res_alt3.csv', sep = '\t', index=False, header=True)
retrieved_res_comb1.to_csv('./scifact/rerank/BM25_res_comb1.csv', sep = '\t', index=False, header=True)
retrieved_res_comb2.to_csv('./scifact/rerank/BM25_res_comb2.csv', sep = '\t', index=False, header=True)
retrieved_res_comb3.to_csv('./scifact/rerank/BM25_res_comb3.csv', sep = '\t', index=False, header=True)




