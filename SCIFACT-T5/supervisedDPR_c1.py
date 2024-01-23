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
train_q1 =  pt.io.read_topics('./scifact/rephrase/T5_train_query1.csv', format='singleline')

train_q2 = train_q
train_q2['query'] = train_q2['query'] + " " + train_q1['query']

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

dict_tr_q2 = dict_query(train_q2)


#train_qrels
new_train_qrels = {}
for i in range(len(df_train_qrels)):
    query_key = str(df_train_qrels['query-id'].iloc[i])
    new_train_qrels[query_key] = {str(df_train_qrels['corpus-id'].iloc[i]) : int(df_train_qrels['score'].iloc[i])}


model_name = "msmarco-distilbert-base-v3"

model = SentenceTransformer(model_name)



retriever = TrainRetriever(model=model, batch_size=16)


train_samples = retriever.load_train(new_corpus, dict_tr_q2, new_train_qrels)

train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
#### training SBERT with dot-product
#train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)


#### Prepare dev evaluator
ir_evaluator = retriever.load_dummy_evaluator()



#### Provide model save path
model_save_path = os.path.join(pathlib.Path('./dpr_models').parent.absolute(), "dpr_models/output_c1", "{}-v1-{}".format(model_name, "SCI-FACT"))
os.makedirs(model_save_path, exist_ok=True)




#### Configure Train params
num_epochs = 100
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)


retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=ir_evaluator,
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)