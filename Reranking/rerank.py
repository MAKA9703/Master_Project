#Import Libraries
import pandas as pd
import os
import ir_datasets
from pygaggle.pygaggle.rerank.base import Query, Text
from pygaggle.pygaggle.rerank.transformer import MonoT5

import logging 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)





logger.info("Loading Data....")
# Load Data
dataset = ir_datasets.load("msmarco-passage")

doc_id = []
doc_text = []

for doc in dataset.docs_iter():
    doc_id.append(doc[0])
    doc_text.append(doc[1])



corpus = pd.DataFrame()
corpus['docno'] = doc_id
corpus['text'] = doc_text


logger.info("Loading Data....")
# Load Data

query_id = []
query_text = []
dataset = ir_datasets.load("msmarco-passage/dev/small")
for query in dataset.queries_iter():
    query_id.append(query[0])
    query_text.append(query[1])

test_query = pd.DataFrame()
test_query['qid'] = query_id
test_query['query'] = query_text



df_test = pd.read_csv('./data/rerank/BM25_res.csv', sep='\t', dtype=str)
#df_test_pseudo = pd.read_csv('./data/rerank/BM25_res_pseudo.csv', sep='\t', dtype=str)

logger.info(df_test)


# Retrieve the top-K documents using BM25
def retrieve_res2(df, docs):
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

def retrieve_res(df, docs):
    passages = []
    for i in range(len(df)):
        if int(df['rank'].iloc[i]) == 0:
            temp = []
            text = docs[docs['docno'] == df['docno'].iloc[i]]
            temp.append([df['docno'].iloc[i], text])

        elif int(df['rank'].iloc[i]) == 199:
            text = docs[docs['docno'] == df['docno'].iloc[i]]
            temp.append([df['docno'].iloc[i],text])
            passages.append(temp)
        
        else:
            text = docs[docs['docno'] == df['docno'].iloc[i]]
            temp.append([df['docno'].iloc[i],text])

        if i % 1000 == 0:
            logger.info(i)
    return passages 



def rerank_func(re_ranker, query, passages, qid):
    final_res = []
    for j in range(len(query)):
        pro_query = Query(query['query'].iloc[j])

        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages[j]]


        reranked = re_ranker.rerank(pro_query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.info(reranked)
        for i in range(len(reranked)):
            str_res = f'{qid.iloc[j]} 0 {reranked[i].metadata["docid"]} {i+1} {reranked[i].score:.5f} T5-ReRank'
            logger.info(str_res)
            final_res.append(str_res)


    return final_res

reranker =  MonoT5('castorini/monot5-large-msmarco-10k')

logger.info('Retrieving Passages:')

retrieved_passages = retrieve_res(df_test, corpus)
#retrieved_passages_pseudo = retrieve_res(df_test_pseudo, corpus)


logger.info("Re-ranking....")

reranked_res = rerank_func(reranker, test_query, retrieved_passages, test_query['qid'])
#reranked_res_pseudo = rerank_func(reranker, test_query, retrieved_passages_pseudo, test_query['qid'])


if os.path.exists('./outputs') == False:
    os.makedirs('./outputs')
with open("./outputs/T5-ReRank.run", "w") as f:
    for l in reranked_res:
        f.write(l + "\n")

#with open("./outputs/T5-ReRank_pseudo.run", "w") as f:
#    for l in reranked_res_pseudo:
#        f.write(l + "\n")


