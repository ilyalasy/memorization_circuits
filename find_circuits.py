
import torch as t
from pathlib import Path
import json

from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType, PatchType, PruneScores
from auto_circuit.utils.graph_utils import patchable_model,edge_counts_util
from auto_circuit.data import load_datasets_from_json
from auto_circuit.utils.tensor_ops import prune_scores_threshold
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune import run_circuits
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.visualize import draw_seq_graph

device = t.device("cuda" if t.cuda.is_available() else "mps")

def find_min_circuit(circuits_metric,eps:float=0.2):
    model_edge_count, base_metric = circuits_metric[-1]
    for edge_count, metric in circuits_metric:
        if metric and (abs(metric - base_metric) / base_metric) < eps:
            return edge_count, metric
    return model_edge_count, base_metric



MODEL_NAME = "EleutherAI/pythia-70m-deduped"
path = Path(f"data/results/contrastive_mem_0.5_pythia-70m-deduped_bleu_ac.json")

with open(path, 'r') as f:
    data = json.load(f)

train_size = int(0.9*len(data["prompts"]))
test_size = len(data["prompts"]) - train_size

model = load_tl_model(MODEL_NAME, device)
model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
    ignore_tokens=[]
)


train_loader, test_loader = load_datasets_from_json(model,path,device,return_seq_length=False,tail_divergence=False,train_test_size=(train_size,test_size),batch_size=8)

path = Path(f"data/circuits/{MODEL_NAME.split('/')[-1]}_mem_prune_scores.pkl")
if path.exists():
    prune_scores: PruneScores = t.load(path, weights_only=False)
else:
    prune_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0, 
        ablation_type=AblationType.RESAMPLE
    )
    t.save(prune_scores, path)

edge_count = edge_counts_util(model.edges, prune_scores=prune_scores)
outs = run_circuits(model,test_loader,edge_count,prune_scores,ablation_type=AblationType.RESAMPLE,patch_type=PatchType.TREE_PATCH)
measurements = measure_answer_diff(model,test_loader,outs)

min_edge_count, min_metric = find_min_circuit(measurements)
threshold = prune_scores_threshold(prune_scores, min_edge_count)
fig = draw_seq_graph(
    model, prune_scores, threshold.item(), layer_spacing=True, orientation="v"
)