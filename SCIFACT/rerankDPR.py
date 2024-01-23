#Import Libraries
import pandas as pd
import os
import pickle 
from pygaggle.pygaggle.rerank.base import Query, Text
from pygaggle.pygaggle.rerank.transformer import MonoT5


print("Loading Data....")
# Load Data
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})


test_queries = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_queries = test_queries[['qid', 'query']]
test_queries = test_queries.drop_duplicates()


with open('./scifact/rerank/dpr_results.pkl', 'rb') as f:
    df_test = pickle.load(f)

for key in df_test:
    for docno, _ in df_test[key].items():
        print(docno)
print("Retrieving results using BM25....")


def retrieve_res(dict_df, docs):
    passages = []
    for key in dict_df:
        temp_list = []
        for docno, _ in dict_df[key].items():
             doc = docs[docs['docno'] == docno]
             text = doc['text'].iloc[0]
             res = [docno, text]
             temp_list.append(res)
        passages.append(temp_list)
    return passages



print("Re-ranking....")

retrieved_passages = retrieve_res(df_test, docs)

reranker =  MonoT5()

def rerank_func(re_ranker, query, passages):
    final_res = []
    qid = query['qid']
    for j in range(len(test_queries)):
        pro_query = Query(query['query'].iloc[j])
        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages[j]]


        reranked = re_ranker.rerank(pro_query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)

        for i in range(25):
            str_res = f'{qid.iloc[j]} 0 {reranked[i].metadata["docid"]} {i+1} {reranked[i].score:.5f} T5-ReRank'
            final_res.append(str_res)


    return final_res

retrieved_passages = retrieve_res(df_test, docs)
reranked_res = rerank_func(reranker, test_queries, retrieved_passages)

if os.path.exists('./outputs') == False:
    os.makedirs('./outputs')
with open("./outputs/T5-ReRankDPR.run", "w") as f:
    for l in reranked_res:
        f.write(l + "\n")

