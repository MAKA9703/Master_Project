import pandas as pd


print("Implementing PyTerrier.....")
import pyterrier as pt
pt.init()


print("Loading Data....")
# Load Corpus
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)

docs = docs.rename(columns={"_id": "docno"})



print("Building Indexes")
# Build DEFAULT index
indexer_both = pt.DFIndexer("./indexes_scifact/both", overwrite=True, blocks=True,verbose=True, stemmer='porter', stopwords='terrier', tokenizer = 'english')



index_both_ref = indexer_both.index(docs["text"], docs["docno"])
index_both = pt.IndexFactory.of(index_both_ref)

print("Both Index Finished!")
