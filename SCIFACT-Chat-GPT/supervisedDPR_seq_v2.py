import pandas as pd


from sentence_transformers import losses, SentenceTransformer
from beir.beir.retrieval.train import TrainRetriever


import pathlib, os

import pyterrier as pt
if not pt.started():
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


#LOAD DATA
corpus = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)

#train data
df_train = pd.read_csv('./scifact/train.csv', sep='\t', dtype=str)
train_q = df_train[['qid', 'query']]


#Rephrased data
train_q1 =  pt.io.read_topics('./scifact/rephrase/CGPT_train_query1.csv', format='singleline')

#train qrels
df_train_qrels = pd.read_csv('./scifact/qrels/train.tsv', sep='\t', dtype=str)



#Rewrite dataframes to dict.
new_corpus = {}
for i in range(len(corpus)):
    key = str(corpus['_id'][i])
    new_corpus[key] = {'text' : corpus['text'][i], 'title' : corpus['title'][i]}

def dict_query(data):
    new_dict = {}
    for i in range(len(data)):
      key = str(data['qid'].iloc[i])
      new_dict[key] = data['query'].iloc[i]

    return new_dict

dict_tr_q = dict_query(train_q)
dict_tr_q1 = dict_query(train_q1)



#train_qrels
new_train_qrels = {}
for i in range(len(df_train_qrels)):
    query_key = str(df_train_qrels['query-id'].iloc[i])
    new_train_qrels[query_key] = {str(df_train_qrels['corpus-id'].iloc[i]) : int(df_train_qrels['score'].iloc[i])}


model_name = "msmarco-distilbert-base-v3"

model_s1 = SentenceTransformer(model_name)
retriever_s1 = TrainRetriever(model=model_s1, batch_size=16)



train_samples_s11 = retriever_s1.load_train(new_corpus, dict_tr_q, new_train_qrels)

train_dataloader_s11 = retriever_s1.prepare_train(train_samples_s11, shuffle=True)


#### Training SBERT with cosine-product
train_loss_s11 = losses.MultipleNegativesRankingLoss(model=retriever_s1.model)

#### training SBERT with dot-product
#train_loss_norm = losses.MultipleNegativesRankingLoss(model=retriever_norm.model, similarity_fct=util.dot_score)



#### Configure Train params
num_epochs = 50
evaluation_steps = 5000
warmup_steps = int(len(train_samples_s11) * num_epochs / retriever_s1.batch_size * 0.1)

ir_evaluator_s11 = retriever_s1.load_dummy_evaluator()

model_save_path_s1 = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_seq_v2", "{}-v1-{}".format(model_name, "SCI-FACT"))
os.makedirs(model_save_path_s1, exist_ok=True)


retriever_s1.fit(train_objectives=[(train_dataloader_s11, train_loss_s11)],
                evaluator=ir_evaluator_s11,
                epochs=num_epochs,
                output_path=model_save_path_s1,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)



train_samples_s12 = retriever_s1.load_train(new_corpus, dict_tr_q1, new_train_qrels)

train_dataloader_s12 = retriever_s1.prepare_train(train_samples_s12, shuffle=True)

#### Training SBERT with cosine-product
train_loss_s12 = losses.MultipleNegativesRankingLoss(model=retriever_s1.model)

ir_evaluator_s12 = retriever_s1.load_dummy_evaluator()



retriever_s1.fit(train_objectives=[(train_dataloader_s12, train_loss_s12)],
                evaluator=ir_evaluator_s12,
                epochs=num_epochs,
                output_path=model_save_path_s1,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)



