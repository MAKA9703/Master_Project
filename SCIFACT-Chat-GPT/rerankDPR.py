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


test_query = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_query = test_query[['qid', 'query']]
test_query = test_query.drop_duplicates()


with open('./scifact/rerank/dpr_results_c1.pkl', 'rb') as f:
    df_test_c1 = pickle.load(f)

with open('./scifact/rerank/dpr_results_c1_alt3.pkl', 'rb') as f:
    df_test_c1_alt3 = pickle.load(f)

with open('./scifact/rerank/dpr_results_c1_comb3.pkl', 'rb') as f:
    df_test_c1_comb3 = pickle.load(f)

with open('./scifact/rerank/dpr_results_c2.pkl', 'rb') as f:
    df_test_c2 = pickle.load(f)

with open('./scifact/rerank/dpr_results_c2_alt3.pkl', 'rb') as f:
    df_test_c2_alt3 = pickle.load(f)

with open('./scifact/rerank/dpr_results_c2_comb3.pkl', 'rb') as f:
    df_test_c2_comb3 = pickle.load(f)

with open('./scifact/rerank/dpr_results_seq_v2.pkl', 'rb') as f:
    df_test_seq = pickle.load(f)

with open('./scifact/rerank/dpr_results_seq_v2_alt3.pkl', 'rb') as f:
    df_test_seq_alt3 = pickle.load(f)

with open('./scifact/rerank/dpr_results_seq_v2_comb3.pkl', 'rb') as f:
    df_test_seq_comb3 = pickle.load(f)


print("Retrieving results using trained DPR model....")


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

retrieved_passages_c1 = retrieve_res(df_test_c1, docs)
retrieved_passages_c2 = retrieve_res(df_test_c2, docs)
retrieved_passages_seq = retrieve_res(df_test_seq, docs)

retrieved_passages_c1_alt3 = retrieve_res(df_test_c1_alt3, docs)
retrieved_passages_c2_alt3 = retrieve_res(df_test_c2_alt3, docs)
retrieved_passages_seq_alt3 = retrieve_res(df_test_seq_alt3, docs)

retrieved_passages_c1_comb3 = retrieve_res(df_test_c1_comb3, docs)
retrieved_passages_c2_comb3 = retrieve_res(df_test_c2_comb3, docs)
retrieved_passages_seq_comb3 = retrieve_res(df_test_seq_comb3, docs)

reranker =  MonoT5('castorini/monot5-large-msmarco-10k')

def rerank_func(re_ranker, query, passages):
    final_res = []
    qid = query['qid']
    for j in range(len(query)):
        pro_query = Query(query['query'].iloc[j])
        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages[j]]

        reranked = re_ranker.rerank(pro_query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)

        for i in range(25):
            str_res = f'{qid.iloc[j]} 0 {reranked[i].metadata["docid"]} {i+1} {reranked[i].score:.5f} T5-ReRank'
            final_res.append(str_res)


    return final_res

def store_res(query, ret_pass, name):
    reranked_res = rerank_func(reranker, query, ret_pass)
    if os.path.exists('./outputs') == False:
        os.makedirs('./outputs')
    with open(f"./outputs/T5-ReRankDPR_{name}.run", "w") as f:
        for l in reranked_res:
            f.write(l + "\n")
        f.close

#C1
store_res(test_query, retrieved_passages_c1, "c1_norm")
store_res(test_query, retrieved_passages_c1_alt3, "c1_alt3_norm")
store_res(test_query, retrieved_passages_c1_comb3, "c1_comb3_norm")


#C2
store_res(test_query, retrieved_passages_c2, "c2_norm")
store_res(test_query, retrieved_passages_c2_alt3, "c2_alt3_norm")
store_res(test_query, retrieved_passages_c2_comb3, "c2_comb3_norm")



#Seq
store_res(test_query, retrieved_passages_seq, "seq_norm")
store_res(test_query, retrieved_passages_seq_alt3, "seq_alt3_norm")
store_res(test_query, retrieved_passages_seq_comb3, "seq_comb3_norm")

