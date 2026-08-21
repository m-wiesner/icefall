#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script takes as input lang_dir and generates HLG from

    - H, the ctc topology, built from tokens contained in lang_dir/lexicon.txt
    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_n_gram.fst.txt

The generated HLG is saved in $lang_dir/HLG.pt
"""
import argparse
import logging
from pathlib import Path

import k2
import torch

from icefall.lexicon import Lexicon
from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm",
        type=str,
        default="G_3_gram",
        help="""Stem name for LM used in HLG compiling.
        """,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )
    parser.add_argument(
        "--modified",
        type=str2bool,
        help="Whether or not to use the modified topo"
    )
    parser.add_argument(
        "--training-topology",
        type=str2bool,
        default=False,
        help="Use the exact sparse topology used by shuffle-CTC training",
    )
    parser.add_argument(
        "--topo-only",
        type=str2bool,
        help="Only the CTC topo."
    )

    return parser.parse_args()


def build_training_ctc_topo(max_token_id: int) -> k2.Fsa:
    """Build the exact sparse CTC topology used by shuffle-CTC training.

    Unlike either variant of :func:`k2.ctc_topo`, this topology requires a
    transition through blank between every pair of emitted tokens.
    """
    final_state = max_token_id + 1
    arcs = ["0 0 0 0 0.0", f"0 {final_state} -1 -1 0.0"]
    for token in range(1, max_token_id + 1):
        arcs.extend(
            (
                f"0 {token} {token} {token} 0.0",
                f"{token} 0 0 0 0.0",
                f"{token} {final_state} -1 -1 0.0",
            )
        )
    arcs.sort(key=lambda arc: int(arc.split()[0]))
    arcs.append(str(final_state))
    return k2.arc_sort(k2.Fsa.from_str("\n".join(arcs), acceptor=False))


def compile_HLG(
    lang_dir: str,
    lm: str = "G_3_gram",
    modified: bool = False,
    training_topology: bool = False,
) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_5000.
      lm:
        The language stem base name.

    Return:
      An FSA representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    if training_topology:
        H = build_training_ctc_topo(max_token_id)
    else:
        H = k2.ctc_topo(max_token_id, modified=modified)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path(f"data/lm/{lm}.pt").is_file():
        logging.info(f"Loading pre-compiled {lm}")
        d = torch.load(f"data/lm/{lm}.pt")
        G = k2.Fsa.from_dict(d)
    else:
        logging.info(f"Loading {lm}.fst.txt")
        with open(f"data/lm/{lm}.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"data/lm/{lm}.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    # LG.labels[LG.labels >= first_token_disambig_id] = 0
    # see https://github.com/k2-fsa/k2/pull/1140
    labels = LG.labels
    labels[labels >= first_token_disambig_id] = 0
    LG.labels = labels

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing H and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"HLG.shape: {HLG.shape}")

    return HLG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    #if not args.modified:
    #    if (lang_dir / "HLG.pt").is_file():
    #        logging.info(f"{lang_dir}/HLG.pt already exists - skipping")
    #        return
    #else:
    #    if (lang_dir / "HLG_modified.pt").is_file():
    #        logging.info(f"{lang_dir}/HLG_modified.pt already exists - skipping")
    #        return


    logging.info(f"Processing {lang_dir}")

    HLG = compile_HLG(
        lang_dir,
        args.lm,
        args.modified,
        getattr(args, "training_topology", False),
    )
    if args.training_topology:
        output = lang_dir / "HLG_training_topology.pt"
    elif args.modified:
        output = lang_dir / "HLG_modified.pt"
    else:
        output = lang_dir / "HLG.pt"
    logging.info(f"Saving HLG to {output}")
    torch.save(HLG.as_dict(), output)

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
