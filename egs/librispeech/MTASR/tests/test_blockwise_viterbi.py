#!/usr/bin/env python3

import unittest

import torch

from shuffle_ctc.blockwise_viterbi import (
    TimedToken,
    add_partial_order_predecessors,
    blockwise_viterbi,
    split_acoustic_blocks,
)


class BlockwiseViterbiTest(unittest.TestCase):
    def test_acoustic_cores_can_be_split_into_smaller_graph_blocks(self):
        token = torch.arange(40).reshape(20, 2)
        speaker = torch.arange(60).reshape(20, 3)
        frames = torch.arange(20)
        local_allowed = torch.ones(20, 4, dtype=torch.bool)
        block = {
            "token_log_probs": token,
            "speaker_log_probs": speaker,
            "absolute_frames": frames,
            "local_token_allowed": local_allowed,
            "diarization_blank_probability": torch.arange(20),
            "diarization_speaker_probability": speaker,
            "diarization_absolute_frames": frames,
            "start": 0.0,
            "end": 20.0,
            "frame_duration": 1.0,
        }
        blocks, maps = split_acoustic_blocks([block], [[2, 0, 1]], 10.0)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(maps, [[2, 0, 1], [2, 0, 1]])
        self.assertEqual(
            [x["absolute_frames"].tolist() for x in blocks],
            [list(range(10)), list(range(10, 20))],
        )
        self.assertEqual([(x["start"], x["end"]) for x in blocks], [
            (0.0, 10.0), (10.0, 20.0)
        ])
        self.assertTrue(torch.equal(
            torch.cat([x["token_log_probs"] for x in blocks]), token
        ))
        self.assertTrue(torch.equal(
            torch.cat([x["diarization_speaker_probability"] for x in blocks]),
            speaker,
        ))

    def test_occurrence_window_is_independent_of_partial_order_collar(self):
        token = TimedToken(
            1, 0, 0, center=10.0, half_width=5.0,
            emission_start=8.0, emission_end=12.0,
        )
        rebuilt = add_partial_order_predecessors([token])[0]
        self.assertEqual(rebuilt.earliest, 8.0)
        self.assertEqual(rebuilt.latest, 12.0)
        self.assertEqual(rebuilt.center, 10.0)
        self.assertEqual(rebuilt.half_width, 5.0)

    def test_two_blocks_match_one_block(self):
        tokens = add_partial_order_predecessors([
            TimedToken(1, 0, 0, 0.4, 0.2),
            TimedToken(2, 1, 1, 1.4, 0.2),
        ])
        token = torch.full((6, 3), -5.0)
        token[:, 0] = -0.1
        token[1, 1] = -0.01
        token[4, 2] = -0.01
        speaker = torch.full((6, 2), -5.0)
        speaker[:3, 0] = -0.01
        # In the second acoustic input, global speaker 1 is local slot 0.
        speaker[3:, 0] = -0.01
        frames = torch.tensor([1, 2, 4, 6, 7, 9])
        common = {"frame_duration": 0.2}
        blocks = [
            dict(common, token_log_probs=token[:3],
                 speaker_log_probs=speaker[:3], absolute_frames=frames[:3],
                 start=0.0, end=1.0),
            dict(common, token_log_probs=token[3:],
                 speaker_log_probs=speaker[3:], absolute_frames=frames[3:],
                 start=1.0, end=2.0),
        ]
        split = blockwise_viterbi(
            tokens, blocks, local_speaker_maps=[[0, 1], [1, 0]]
        )
        whole = blockwise_viterbi(tokens, [
            dict(common, token_log_probs=token,
                 speaker_log_probs=torch.cat((speaker[:3], speaker[3:].flip(-1))),
                 absolute_frames=frames, start=0.0, end=2.0)
        ], local_speaker_maps=[[0, 1]])
        self.assertEqual(split.token_indexes.tolist(), [0, 1])
        self.assertEqual(split.frames.tolist(), whole.frames.tolist())
        self.assertEqual(split.frames.tolist(), [2, 7])
        self.assertEqual(split.boundary_state_counts, [1, 1, 1])

    def test_shuffle_only_ignores_speaker_scores(self):
        tokens = add_partial_order_predecessors([
            TimedToken(1, 0, 0, 0.2, 1.0),
            TimedToken(2, 1, 1, 0.6, 1.0),
        ])
        token = torch.tensor([
            [-0.1, -0.01, -5.0],
            [-0.1, -5.0, -0.01],
        ])
        # Deliberately favor the wrong speaker at both frames.
        speaker = torch.tensor([[-10.0, 0.0], [0.0, -10.0]])
        result = blockwise_viterbi(
            tokens,
            [{"token_log_probs": token, "speaker_log_probs": speaker,
              "absolute_frames": torch.tensor([0, 1]), "start": 0.0,
              "end": 1.0, "frame_duration": 0.4}],
            local_speaker_maps=[[0, 1]], score_speakers=False,
        )
        self.assertEqual(result.token_indexes.tolist(), [0, 1])

    def test_training_and_original_topologies(self):
        # Two identical consecutive labels require a blank in both topologies;
        # the original topology may additionally repeat a label without output.
        tokens = add_partial_order_predecessors([
            TimedToken(1, 0, 0, 0.2, 1.0),
            TimedToken(1, 0, 1, 0.8, 1.0),
        ])
        token = torch.tensor([
            [-5.0, -0.01], [-0.01, -5.0], [-5.0, -0.01]
        ])
        block = {"token_log_probs": token,
                 "speaker_log_probs": torch.zeros(3, 1),
                 "absolute_frames": torch.tensor([0, 1, 2]),
                 "start": 0.0, "end": 1.2, "frame_duration": 0.4}
        for topology in ("training", "original"):
            result = blockwise_viterbi(
                tokens, [block], local_speaker_maps=[[0]], topology=topology,
            )
            self.assertEqual(result.token_indexes.tolist(), [0, 1])
            self.assertEqual(result.frames.tolist(), [0, 2])

    def test_pruned_mode(self):
        tokens = add_partial_order_predecessors([
            TimedToken(1, 0, 0, 0.2, 1.0),
        ])
        block = {"token_log_probs": torch.tensor([[-1.0, -0.01]]),
                 "speaker_log_probs": torch.zeros(1, 1),
                 "absolute_frames": torch.tensor([0]), "start": 0.0,
                 "end": 1.0, "frame_duration": 0.4}
        result = blockwise_viterbi(
            tokens, [block], local_speaker_maps=[[0]], search_beam=1.0,
            min_active_states=0, max_active_states=10,
        )
        self.assertEqual(result.token_indexes.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
