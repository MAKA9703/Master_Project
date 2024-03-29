#Import Libraries
import pandas as pd
import pytrec_eval
import glob

#qrels
test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)
test_qrels = test_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
test_qrels['iteration'] = 0
test_qrels = test_qrels.astype({'label': 'int32'})

# Load qrels in a dictionary
qrels_dict = dict()
for _, r in test_qrels.iterrows():
    qid, docno, label, iteration = r
    if qid not in qrels_dict:
        qrels_dict[qid] = dict()
    qrels_dict[qid][docno] = int(label)

# Build evaluator based on the qrels and metrics
metrics = {"ndcg_cut_5", "ndcg_cut_10", "map_cut_10", "recall_10", "recip_rank"}
my_qrel = {q: d for q, d in qrels_dict.items()}
evaluator = pytrec_eval.RelevanceEvaluator(my_qrel, metrics)

files = glob.glob("./outputs/*.run")



def evaluate_run(run, name):
    print(f"Results for {name}")
    # Evaluate model
    evals = evaluator.evaluate(run)

    # Compute performance in different metrics for each query
    metric2vals = {m: [] for m in metrics}
    for q, d in evals.items():
        for m, val in d.items():
            metric2vals[m].append(val)
    
    # Average results by query
    metric2avg = dict()
    for m in metrics:
        val = pytrec_eval.compute_aggregated_measure(m, metric2vals[m])
        metric2avg[m] = val
        print(m, '\t', val)

def print_eval_run(files):
    # Load runs
    for file in files:
        with open(file, 'r') as f:
            run = pytrec_eval.parse_run(f)
        evaluate_run(run, file)

print_eval_run(files)
