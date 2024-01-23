#Import Libraries
import pandas as pd
import os
from pygaggle.pygaggle.rerank.base import Query, Text
from pygaggle.pygaggle.rerank.transformer import MonoT5





print("Loading Data....")
# Load Data
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})


df_test = pd.read_csv('./scifact/rerank/BM25_res.csv', sep='\t', dtype=str)
test_queries = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_queries = test_queries[['qid', 'query']]
test_queries = test_queries.drop_duplicates()



print("Retrieving results using BM25....")

# Retrieve the top-K documents using BM25
def retrieve_res(df, docs):
    passages = []
    temp = '-1' 
    temp_list = []
    for i in range(len(df)):
        key = str(df['docno'].iloc[i])
        doc = docs[docs['docno'] == key]
        text = doc['text'].iloc[0]
        if str(df['qid'].iloc[i])  == str(temp) and i != (len(df)-1):
            res = [key, text]
            temp_list.append(res)
        elif str(df['qid'].iloc[i]) != str(temp) and i != 0:
            passages.append(temp_list)
            temp = df['qid'].iloc[i]
            temp_list = []
            res = [key, text]
            temp_list.append(res)
        elif i == 0:
            res = [key, text]
            temp_list.append(res)
            temp = df['qid'].iloc[i]
        else: 
            res = [key, text]
            temp_list.append(res)
            passages.append(temp_list)
    return passages

print("Re-ranking....")

def rerank_func(re_ranker, query, passages, qid):
    final_res = []
    for j in range(len(test_queries)):
        pro_query = Query(query['query'].iloc[j])

        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages[j]]


        reranked = re_ranker.rerank(pro_query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)

        for i in range(25):
            str_res = f'{qid.iloc[j]} 0 {reranked[i].metadata["docid"]} {i+1} {reranked[i].score:.5f} T5-ReRank'
            print(str_res)
            final_res.append(str_res)


    return final_res

reranker =  MonoT5('castorini/monot5-large-msmarco-10k')

retrieved_passages = retrieve_res(df_test, docs)
reranked_res = rerank_func(reranker, test_queries, retrieved_passages, test_queries['qid'])

if os.path.exists('./outputs') == False:
    os.makedirs('./outputs')
with open("./outputs/T5-ReRank.run", "w") as f:
    for l in reranked_res:
        f.write(l + "\n")

