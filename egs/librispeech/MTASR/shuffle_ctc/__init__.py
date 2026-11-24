"""Reusable k2-based losses for CTC over shuffle-constraint graphs."""

from .loss import GraphCTCLoss, ShuffleCTCLoss, graph_ctc_loss
from .factorized import factorized_graph_ctc_loss, intersect_dense_factorized
from .viterbi import (
    ViterbiAlignment,
    factorized_viterbi,
    factorized_viterbi_batched,
    token_viterbi_batched,
)
from .blockwise_viterbi import (
    BlockwiseAlignment, TimedToken, add_partial_order_predecessors,
    blockwise_viterbi, split_acoustic_blocks,
)

__all__ = [
    "GraphCTCLoss", "ShuffleCTCLoss", "graph_ctc_loss",
    "factorized_graph_ctc_loss", "intersect_dense_factorized",
    "ViterbiAlignment", "factorized_viterbi", "factorized_viterbi_batched",
    "token_viterbi_batched",
    "BlockwiseAlignment", "TimedToken", "add_partial_order_predecessors",
    "blockwise_viterbi", "split_acoustic_blocks",
]
