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
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})

#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test2 = df_test[['qid', 'query']]
df_test2.to_csv('./scifact/BM25_test_queries.csv', sep = '\t', index=False, header=False)
test_query = pt.io.read_topics('./scifact/BM25_test_queries.csv', format='singleline')
test_query = test_query.drop_duplicates()



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





retrieved_res.to_csv('./scifact/rerank_test_res.csv', sep = '\t', index=False, header=True)




