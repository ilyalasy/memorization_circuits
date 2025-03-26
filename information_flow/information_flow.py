# Based on https://github.com/facebookresearch/llm-transparency-tool

from typing import List, Optional

from fancy_einsum import einsum
import networkx as nx
import torch
from torch.utils.data import DataLoader
import information_flow.contributions as contributions
from transformer_lens import HookedTransformer
from tqdm import tqdm

class GraphBuilder:
    """
    Constructs the contributions graph with edges given one by one. The resulting graph
    is a networkx graph that can be accessed via the `graph` field. It contains the
    following types of nodes:

    - X0_<token>: the original token.
    - A<layer>_<token>: the residual stream after attention at the given layer for the
        given token.
    - M<layer>_<token>: the ffn block.
    - I<layer>_<token>: the residual stream after the ffn block.
    """

    def __init__(self, n_layers: int, n_tokens: int):
        self._n_layers = n_layers
        self._n_tokens = n_tokens

        self.graph = nx.DiGraph()
        for layer in range(n_layers):
            for token in range(n_tokens):
                self.graph.add_node(f"A{layer}_{token}")
                self.graph.add_node(f"I{layer}_{token}")
                self.graph.add_node(f"M{layer}_{token}")
        for token in range(n_tokens):
            self.graph.add_node(f"X0_{token}")
        self.graph.graph['n_layers'] = n_layers
        self.graph.graph['n_tokens'] = n_tokens

    def get_output_node(self, token: int):
        return f"I{self._n_layers - 1}_{token}"

    def _add_edge(self, u: str, v: str, weight: float):
        # TODO(igortufanov): Here we sum up weights for multi-edges. It happens with
        # attention from the current token and the residual edge. Ideally these need to
        # be 2 separate edges, but then we need to do a MultiGraph. Multigraph is fine,
        # but when we try to traverse it, we face some NetworkX issue with EDGE_OK
        # receiving 3 arguments instead of 2.
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] += weight
        else:
            self.graph.add_edge(u, v, weight=weight)

    def add_attention_edge(self, layer: int, token_from: int, token_to: int, w: float):
        self._add_edge(
            f"I{layer-1}_{token_from}" if layer > 0 else f"X0_{token_from}",
            f"A{layer}_{token_to}",
            w,
        )

    def add_residual_to_attn(self, layer: int, token: int, w: float, parallel: bool = False):
        self._add_edge(
            f"I{layer-1}_{token}" if layer > 0 else f"X0_{token}",
            f"A{layer}_{token}",
            w,
        )
        if parallel:
            self._add_edge(f"A{layer}_{token}", f"I{layer}_{token}", w)

    def add_ffn_edge(self, layer: int, token: int, w: float, parallel: bool = False):
        if not parallel:
            self._add_edge(f"A{layer}_{token}", f"M{layer}_{token}", w)
        else:
            self._add_edge(f"I{layer-1}_{token}" if layer > 0 else f"X0_{token}", f"M{layer}_{token}", w)
        self._add_edge(f"M{layer}_{token}", f"I{layer}_{token}", w)

    def add_residual_to_ffn(self, layer: int, token: int, w: float, parallel: bool = False):
        if not parallel:
            self._add_edge(f"A{layer}_{token}", f"I{layer}_{token}", w)
        else:
            self._add_edge(f"I{layer-1}_{token}" if layer > 0 else f"X0_{token}", f"I{layer}_{token}", w)

def decomposed_attn_batch(
        model: HookedTransformer, cache:dict, layer: int
    ) -> torch.Tensor:
        hook_v:torch.Tensor = cache[f"blocks.{layer}.attn.hook_v"]
        b_v:torch.Tensor = model.blocks[layer].attn.b_V

        # support for gqa
        num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)

        v = hook_v + b_v
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"].to(v.dtype)
        z = einsum(
            "batch key_pos head d_head, "
            "batch head query_pos key_pos -> "
            "batch query_pos key_pos head d_head",
            v,
            pattern,
        )
        decomposed_attn = einsum(
            "batch pos key_pos head d_head, "
            "head d_head d_model -> "
            "batch pos key_pos head d_model",
            z,
            model.blocks[layer].attn.W_O,
        )
        return decomposed_attn

@torch.no_grad()
def build_full_graph(
    model: HookedTransformer,
    dataloader: DataLoader,
    renormalizing_threshold: Optional[float] = None,    
) -> nx.Graph:
    """
    Build the contribution graph for all blocks of the model and all tokens.

    model: The transparent llm which already did the inference.
    batch_i: Which sentence to use from the batch that was given to the model.
    renormalizing_threshold: If specified, will apply renormalizing thresholding to the
    contributions. All contributions below the threshold will be erazed and the rest
    will be renormalized.
    """
    n_layers = model.cfg.n_layers
    parallel = model.cfg.parallel_attn_mlp

    graphs = []
    for batch_i, (batch,_) in enumerate(dataloader):        
        logits, cache = model.run_with_cache(batch.to(model.cfg.device))

        for i, sample in tqdm(enumerate(batch),desc=f"Creating graphs for batch {batch_i+1}/{len(dataloader)}", total=len(batch)):
            n_tokens = sample.size(0)

            builder = GraphBuilder(n_layers, n_tokens)

            for layer in range(n_layers):
                resid_out = cache[f"blocks.{layer}.hook_resid_post"][i].unsqueeze(0)
                mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][i].unsqueeze(0) 
                resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][i].unsqueeze(0)
                
                decomposed_attn = decomposed_attn_batch(model, cache, layer)[i].unsqueeze(0)
                if parallel:                                
                    decomposed_attn = decomposed_attn_batch(model, cache, layer)[i].unsqueeze(0)
                    c_attn, c_ffn, c_resid = contributions.get_parallel_attention_contributions(
                        resid_pre,
                        decomposed_attn,
                        mlp_out,
                        resid_out,
                    )                               

                    if renormalizing_threshold is not None:  
                        c_attn, c_ffn, c_resid = contributions.apply_threshold_and_renormalize_parallel(
                            renormalizing_threshold, c_attn, c_ffn, c_resid
                        )
                    
                    c_resid_attn = c_resid
                    c_resid_ffn = c_resid
                else:                    
                    resid_mid = cache[f"blocks.{layer}.hook_resid_mid"][i].unsqueeze(0)
                    c_attn, c_resid_attn = contributions.get_attention_contributions(
                        resid_pre=resid_pre, # [1,13,768]
                        resid_mid=resid_mid, # same
                        decomposed_attn=decomposed_attn, # [1,13,13,12,768]
                    )

                    c_ffn, c_resid_ffn = contributions.get_mlp_contributions(
                        resid_mid=resid_mid,## [1,13,768]
                        resid_post=resid_out,## [1,13,768]
                        mlp_out=mlp_out, ## [1,13,768]
                    )

                    if renormalizing_threshold is not None:
                        c_attn, c_resid_attn = contributions.apply_threshold_and_renormalize(
                            renormalizing_threshold, c_attn, c_resid_attn
                        )
                        c_ffn, c_resid_ffn = contributions.apply_threshold_and_renormalize(
                            renormalizing_threshold, c_ffn, c_resid_ffn
                        )


                for token_from in range(n_tokens):
                    for token_to in range(n_tokens):
                        # Sum attention contributions over heads.
                        c = c_attn[0, token_to, token_from].sum().item()
                        builder.add_attention_edge(layer, token_from, token_to, c)
                for token in range(n_tokens):
                    builder.add_residual_to_attn(
                        layer, token, c_resid_attn[0, token].item(), parallel
                    )

                for token in range(n_tokens):
                    builder.add_ffn_edge(layer, token, c_ffn[0, token].item(), parallel)
                    builder.add_residual_to_ffn(
                        layer, token, c_resid_ffn[0, token].item(), parallel
                    )      
            graphs.append(builder.graph)
    return graphs


def find_longest_contribution_path(
    graph: nx.Graph,
    input_token: int,
    output_token: int,
    threshold: float = 0.0
) -> tuple[List[str], float]:
    """
    Finds the path with highest total contribution weight from an input token to an output token
    using NetworkX's dag_longest_path function.
    
    Args:
        graph: The contribution graph
        input_token: Index of the input token
        output_token: Index of the output token
        threshold: Minimum edge weight to consider (default: 0.0)
    
    Returns:
        tuple containing:
        - List of node names representing the path
        - Total weight of the path
    """
    # Create view of graph with only edges above threshold
    filtered_graph = nx.subgraph_view(
        graph,
        filter_edge=lambda u, v: graph[u][v]["weight"] > threshold
    )
    
    source = f"X0_{input_token}"
    target = f"I{graph.graph['n_layers']-1}_{output_token}"
    
    # Create a subgraph between source and target
    reachable_nodes = nx.descendants(filtered_graph, source) | {source}
    ancestors_of_target = nx.ancestors(filtered_graph, target) | {target}
    relevant_nodes = reachable_nodes & ancestors_of_target
    
    if not relevant_nodes:
        return [], 0.0
        
    subgraph = filtered_graph.subgraph(relevant_nodes)
    
    try:
        # Find longest path in the DAG using edge weights
        path = nx.dag_longest_path(
            subgraph,
            weight='weight',
            default_weight=0.0
        )
        
        # Calculate total weight
        total_weight = sum(
            filtered_graph[path[i]][path[i+1]]['weight']
            for i in range(len(path)-1)
        )
        
        return path, total_weight
        
    except (nx.NetworkXNoPath, nx.NetworkXError):
        return [], 0.0

def build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    threshold: float,
    starting_tokens: Optional[List[int]] = None,
) -> List[nx.Graph]:
    """
    Given the full graph, this function returns only the trees leading to the specified
    tokens. Edges with weight below `threshold` will be ignored. 
    """
    if starting_tokens is None:
        starting_tokens = list(range(n_tokens))
    builder = GraphBuilder(n_layers, n_tokens)

    rgraph = graph.reverse()
    search_graph = nx.subgraph_view(
        rgraph, filter_edge=lambda u, v: rgraph[u][v]["weight"] > threshold
    )

    result = []
    for start in starting_tokens:
        assert start < n_tokens
        assert start >= 0
        edges = nx.edge_dfs(search_graph, source=builder.get_output_node(start))
        tree = search_graph.edge_subgraph(edges)
        # Reverse the edges because the dfs was going from upper layer downwards.
        result.append(tree.reverse())

    return result


def get_best_tokens(
    graph: nx.Graph,        
    threshold: float,
) -> dict[int, float]:
    """
    Given calculate contribution graph, return the tokens with the highest contribution to the output.
    """
    n_layers = graph.graph['n_layers']
    n_tokens = graph.graph['n_tokens']

    # Get the contribution paths to the specified output tokens

    token_contributions = {i: 0.0 for i in range(n_tokens)}
    output_token = n_tokens - 1
    for input_token in range(n_tokens):
        path, contribution = find_longest_contribution_path(
            graph, input_token, output_token, threshold
        )
        token_contributions[input_token] = contribution
    
    # Sort tokens by their contribution in descending order and return as a dictionary
    sorted_tokens_dict = {
        token: contribution for token, contribution in sorted(
            token_contributions.items(),
            key=lambda item: item[1],
            reverse=True
        )
    }
    
    return sorted_tokens_dict