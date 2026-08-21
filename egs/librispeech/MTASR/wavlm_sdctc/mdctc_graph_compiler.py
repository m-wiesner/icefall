#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Matthew Wiesner
# Year: 2025

import argparse
import k2
import sentencepiece as spm
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
import torch
from collections import defaultdict, deque
from itertools import groupby
import logging


class MDCTCGraphCompiler(object):
    """
    A graph compiler for multi-dimensional CTC decoding with token-level timing constraints.

    This class is designed to construct decoding graphs (FSAs) for use in alignment or training
    under the Multi-Dimensional CTC (MD-CTC) framework. It uses a SentencePiece model to map
    transcripts to token sequences and can encode optional time-collar constraints between tokens.

    Attributes
    ----------
    sp : sentencepiece.SentencePieceProcessor
        The SentencePiece model used to tokenize input strings.
    device : Union[str, torch.device]
        The device on which FSAs will be created and processed (e.g., "cpu" or "cuda").
    collar : int
        A temporal padding (in frames or milliseconds) to apply as a constraint margin between tokens.
    """
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
    ):
        """
            Initialize the MDCTCGraphCompiler.

            Loads the SentencePiece model from the given language directory and sets up
            internal parameters such as device and collar width.

            :param lang_dir: Path to the language directory containing the `bpe.model` file.
            :type lang_dir: Path
            :param device: The device to use for FSA operations (e.g., "cpu" or "cuda").
            :type device: Union[str, torch.device]
            :param collar: Temporal collar size (e.g., in milliseconds or frames) to allow flexibility in alignment.
            :type collar: int
        """
        self.device = device
        # Create a map from (speaker, token) tuples --> int
        lang_dir = Path(lang_dir)
        bpe_model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor() 
        sp.load(str(bpe_model_file))
        self.sp = sp


    def build_ctc_topo(self, symbols: List[int], original=False) -> k2.Fsa:
        """
            Builds a CTC topology over the provided symbols only.

            :param symbols: The list of the symbols over which to build the 
                            ctc topology. 0 is assumed to be blank on the input
                            and epsilon on the output.
        """
        assert 0 not in symbols, "Symbol 0 is reserved for the blank token in CTC."
        
        symbols = list(set(symbols))
        arcs = []
        blank_state = 0  # Will set this properly later
        arcs.append(f"{blank_state} {blank_state} 0 0 0.0")  # self-loop with blank
        arcs.append(f"{blank_state} {len(symbols)+1} -1 -1 0.0")
        for i, s1 in enumerate(symbols, 1):
            # Blank state:
            #  - self-loop with blank
            #  - arc to symbol state with symbol label
            arcs.append(f"{blank_state} {i} {s1} {s1} 0.0")    # to symbol state
        
            # only self-loop and go to blank
            arcs.append(f"{i} 0 0 0 0.0")  # to blank
            if original:
                arcs.append(f"{i} {i} {s1} 0 0.0")     # self-loop with symbol
                for j, s2 in enumerate(symbols, 1):
                    if i == j:
                        continue
                    arcs.append(f"{i} {j} {s2} {s2} 0.0")
            arcs.append(f"{i} {len(symbols)+1} -1 -1 0.0")
        arcs = sorted(arcs, key=lambda x: int(x.split()[0]))
        arcs.append(f"{len(symbols)+1}")
        # Build arc list for k2
        fsa = k2.Fsa.from_str("\n".join(arcs), acceptor=False)
        
        fsa = k2.arc_sort(fsa)
        return fsa 

    def compile(
        self,
        cuts: List[List[str]],
        original_topo: bool = False,
        debug: bool = False,
    ) -> k2.Fsa:
        """
            Compile a batch of transcripts into CTC-constrained decoding graphs.

            This function encodes each transcript into subword/token IDs, constructs a
            constrained alignment FSA based on per-token constraints, and composes it
            with a CTC topology graph to produce a training graph. The result is a
            `k2.FsaVec` of decoding graphs suitable for CTC training.

            :param cuts: A batch of transcripts, where each transcript is a list of string tokens.
            :type cuts: List[List[str]]
            :param original_topo: Which ctc topology to use
            :type original_topo: bool
            :return: A batched FSA (`FsaVec`) representing all input transcripts composed with a CTC topology.
            :rtype: k2.Fsa
        """
        graphs = []
        for c in cuts:
            labels = self.sp.encode(c)
            
            fsa = k2.linear_fsa(labels)       
            fsa = fsa.to(self.device)
            
            ctc_topo = self.build_ctc_topo(
                labels,
                original=original_topo
            )
           
            ctc_topo = ctc_topo.to(self.device)
            fsa_with_self_loop = k2.remove_epsilon_and_add_self_loops(fsa)
            fsa_with_self_loop = k2.connect(fsa_with_self_loop)
            fsa_with_self_loop = k2.arc_sort(fsa_with_self_loop)
            graph = k2.compose(
                ctc_topo,
                fsa_with_self_loop,
                treat_epsilons_specially=False,
            )
            graphs.append(graph)
        training_graphs = k2.create_fsa_vec(graphs)
        return training_graphs

    
def main(args):
    
    # Your code here
    #test_make_transcript_fsa()
    #test_compose_speaker_transcripts(args)
    #test_make_multispeaker_supervision_graph(args)
    import time
    start = time.time()
    test_compile(args)
    print(f"Elapsed: {time.time() - start}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The MDCTC Compiler")
    # parser.add_argument('--example', help='An example argument')
    parser.add_argument(
        "--langdir",
        help="The path to the directory containing the bpe.model file created"
             " by the sentencepiece module.",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)

