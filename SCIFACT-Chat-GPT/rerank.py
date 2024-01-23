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
df_test_alt3 = pd.read_csv('./scifact/rerank/BM25_res_alt3.csv', sep='\t', dtype=str)
df_test_comb3 = pd.read_csv('./scifact/rerank/BM25_res_comb3.csv', sep='\t', dtype=str)

test_query = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_query = test_query[['qid', 'query']]
test_query = test_query.drop_duplicates()


#test_data 
test_query1 = pd.DataFrame()
test_query1[['qid','query']] = pd.read_csv('./scifact/rephrase/CGPT_test_query1.csv', header=None, sep='\t', dtype=str)
test_query1 = test_query1.drop_duplicates(subset=['qid'])

test_query2 = pd.DataFrame()
test_query2[['qid','query']] = pd.read_csv('./scifact/rephrase/CGPT_test_query2.csv', header=None, sep='\t', dtype=str)
test_query2 = test_query2.drop_duplicates(subset=['qid'])

test_query3 = pd.DataFrame()
test_query3[['qid','query']] = pd.read_csv('./scifact/rephrase/CGPT_test_query3.csv', header=None, sep='\t', dtype=str)
test_query3 = test_query3.drop_duplicates(subset=['qid'])



test_alt3 = pd.DataFrame()
test_alt3['qid'] = test_query['qid']
test_alt3['query'] = test_query['query'] + ' ' + test_query1['query'] + ' ' + test_query['query'] + ' ' + test_query2['query'] + ' ' + test_query['query'] + ' ' + test_query3['query']  
test_alt3 = test_alt3.astype({'qid': 'string', 'query':'string'})

test_comb3 = pd.DataFrame()
test_comb3['qid'] = test_query['qid']
test_comb3['query'] = test_query['query'] + ' ' + test_query1['query'] + ' ' + test_query2['query'] + ' ' + test_query3['query'] 
test_comb3 = test_comb3.astype({'qid': 'string', 'query':'string'})

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
    for j in range(len(query)):
        pro_query = Query(query['query'].iloc[j])

        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages[j]]


        reranked = re_ranker.rerank(pro_query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)

        for i in range(25):
            str_res = f'{qid.iloc[j]} 0 {reranked[i].metadata["docid"]} {i+1} {reranked[i].score:.5f} T5-ReRank'
            final_res.append(str_res)


    return final_res

reranker =  MonoT5('castorini/monot5-large-msmarco-10k')

def save_rerank(queries, docs, bm25_res, name):
    retrieved_passages = retrieve_res(bm25_res, docs)
    reranked_res = rerank_func(reranker, queries, retrieved_passages, queries['qid'])

    if os.path.exists('./outputs') == False:
        os.makedirs('./outputs')
    with open(f"./outputs/T5-ReRank_{name}.run", "w") as f:
        for l in reranked_res:
            f.write(l + "\n")


save_rerank(test_query, docs, df_test, "norm")
save_rerank(test_alt3, docs, df_test, "norm_alt3")
save_rerank(test_comb3, docs, df_test, "norm_comb3")

save_rerank(test_query, docs, df_test_alt3, "alt3")
save_rerank(test_alt3, docs, df_test_alt3, "alt3_alt3")
save_rerank(test_comb3, docs, df_test_alt3, "alt3_comb3")

save_rerank(test_query, docs, df_test_comb3, "comb3")
save_rerank(test_alt3, docs, df_test_comb3, "comb3_alt3")
save_rerank(test_comb3, docs, df_test_comb3, "comb3_comb3")
