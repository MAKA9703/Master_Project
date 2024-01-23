import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


from openai import OpenAI
client = OpenAI(api_key='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')


#train data
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
test_source = df_test['query']


def paraphrase_query(query):
  paraphrased_queries = [] 
  for _ in range(3):
    model = 'gpt-3.5-turbo'
    response = client.chat.completions.create(
      model = model,
      #prompt=f"Paraphrase the following query: {query}",
      messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Paraphrase the following query: {query}"},
  ],
      temperature = 0.7,
      max_tokens=256
    )
    paraphrased_queries.append(response.choices[0].message.content)
  #return the paraphrased queries
  return paraphrased_queries




#Use model to paraphrase each of the queries in the training data-set.
all_test_outputs = []
for i in range(len(test_source)):
  res = paraphrase_query(test_source.iloc[i])
  all_test_outputs.append(res)
  if i < 5: 
    print("Original Query: ", test_source.iloc[i])
    print("Paraphrased Queries: ", res)

tr1 = []
tr2 = []
tr3 = []
for i in range(len(all_test_outputs)):
  tr1.append(all_test_outputs[i][0])
  tr2.append(all_test_outputs[i][1])
  tr3.append(all_test_outputs[i][2])


train_new_queries1 = pd.DataFrame()
train_new_queries1['qid'] = df_test['qid']
train_new_queries1['query'] = tr1

train_new_queries2 = pd.DataFrame()
train_new_queries2['qid'] = df_test['qid']
train_new_queries2['query'] = tr2

train_new_queries3 = pd.DataFrame()
train_new_queries3['qid'] = df_test['qid']
train_new_queries3['query'] = tr3



#Save files
train_new_queries1.to_csv('./scifact/rephrase/my_test_query1.csv', sep='\t', index = False, header = False)
train_new_queries2.to_csv('./scifact/rephrase/my_test_query2.csv', sep='\t', index = False, header = False)
train_new_queries3.to_csv('./scifact/rephrase/my_test_query3.csv', sep='\t', index = False, header = False)

