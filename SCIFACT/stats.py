import pandas as pd
import numpy as np

print("Implementing PyTerrier.....")
#Implement PyTerrier
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

print("Loading Data....")
# Load Data
#documents
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})

#train_data
train_query = pd.read_csv('./scifact/train.csv', sep='\t', dtype=str)


#test_data 
test_query = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)



#qrels
train_qrels = pd.read_csv('./scifact/qrels/train.tsv', sep='\t', dtype=str)
train_qrels = train_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
train_qrels['iteration'] = 0

test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)
test_qrels = test_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
test_qrels['iteration'] = 0


print("Loading already build index for documents.....")
# Loading already build index
index_ref_both = pt.IndexRef.of("./indexes/both/data.properties")
index_both = pt.IndexFactory.of(index_ref_both)


print('Document Stats:')
print('')
print(index_both.getCollectionStatistics().toString())
print('')
print('')
print('General Train Query Stats:')
print('')
print('')
train_query['doc_len'] = train_query['query'].map(len)
print(train_query['doc_len'].describe())
print('Median:', train_query['doc_len'].median())
print('Total #Queries:', len(train_query))
print('')
print('')
print('General Test Query Stats:')
print('')
print('')
test_query['doc_len'] = test_query['query'].map(len)
print(test_query['doc_len'].describe())
print('Median:', test_query['doc_len'].median())
print('Total #Queries:', len(test_query))
print('')
print('')
print('More General Corpus Stats:')
print('')
print('')
print('Default Corpus:')
print('')

doc_lengths2 = []
for i in range(index_both.getDocumentIndex().getNumberOfDocuments()):
    doc_lengths2.append(index_both.getDocumentIndex().getDocumentLength(i))

print('')
print('')
print('Both Corpus:')
print('')
print('Minimum Length of Corpus:', np.min(doc_lengths2))
print('Maximum Length of Corpus:', np.max(doc_lengths2))
print('Average Length of Corpus:', np.mean(doc_lengths2))
print('Median Length of Corpus:', np.median(doc_lengths2))
print('')
print('')
print('MISSING:')
print('Distribution Plots')
print('Search Time Results')