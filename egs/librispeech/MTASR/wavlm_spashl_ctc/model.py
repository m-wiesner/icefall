# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def split_joint_labels(
    labels: torch.Tensor, token_vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert ``token + speaker * (V - 1)`` for positive joint labels."""
    if token_vocab_size < 2:
        raise ValueError(f"token_vocab_size must be at least 2, got {token_vocab_size}")
    if labels.numel() and bool((labels <= 0).any().item()):
        raise ValueError("split_joint_labels expects nonblank labels (> 0)")
    block_size = token_vocab_size - 1
    speakers = torch.div(labels - 1, block_size, rounding_mode="floor")
    tokens = labels - speakers * block_size
    return speakers, tokens


class MDCTCModel(nn.Module):
    """MD-CTC model with separate token and speaker output heads."""

    def __init__(
        self,
        encoder,
        encoder_dim: int,
        vocab_size1: int,
        vocab_size2: int,
        freeze_feat_extractor: bool = True,
        factorized_token_speaker_output: bool = False,
        layer_norm: bool = False,
        downsample: bool = True,
    ):
        """
        Args:
          encoder:
            The shared encoder for the ShuffleCTC and diarization
            branches
          encoder_dim:
            Dimension of the encoder output.
          vocab_size:
            Number of tokens of the modeling unit including blank.
        """
        super().__init__()
        # Freeze the feature extractor (CNN frontend)
        try:
            for param in encoder.feature_extractor.parameters():
                param.requires_grad = False
        except AttributeError:
            for param in encoder.model.feature_extractor.parameters():
                param.requires_grad = False
        self.freeze_feat_extractor = freeze_feat_extractor
        self.frozen = False
        self.encoder = encoder
        self.ctc_output1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size1),
        ) 

        self.ctc_output2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size2),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.factorized_token_speaker_output = factorized_token_speaker_output
        self.use_layer_norm = layer_norm
        self.downsample = downsample

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        speaker_weight: float = 1.0,
        return_factors: bool = False,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          Returns:
          Return the CTC output and the output lengths.
        """
        assert x_lens.ndim == 1, x_lens.shape
        nnet_output = self.encoder(x)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        assert torch.all(x_lens > 0)
        
        # We will upsample here to handle overlapping speech
        #nnet_output, x_lens = self.upsampler(nnet_output, x_lens)
        #nnet_output = self.upsampler(
        #    nnet_output.transpose(1, 2)
        #).transpose(1, 2) 
        #x_lens = x_lens * 2

        if self.downsample:
            nnet_output = nnet_output.transpose(1, 2)
            nnet_output = nn.functional.avg_pool1d(nnet_output, kernel_size=2, stride=2)
            nnet_output = nnet_output.transpose(1, 2)
            x_lens = torch.floor((x_lens - 2) / 2 + 1).to(torch.int32)
        
        if self.use_layer_norm:
            nnet_output = self.layer_norm(nnet_output)
        
        # compute ctc log-probs
        ctc_output1 = self.ctc_output1(nnet_output)
        ctc_output2 = self.ctc_output2(nnet_output)
        if self.factorized_token_speaker_output:
            out1 = self.log_softmax(ctc_output1)
            blank = out1[..., 0]
            out2 = speaker_weight * self.log_softmax(ctc_output2)
            if return_factors:
                return out1, out2, x_lens
            out = out1[..., 1:].unsqueeze(-1) + out2.unsqueeze(-2)
            out = out.permute(0, 1, 3, 2)
            out = out.reshape(out.size(0), out.size(1), -1)
            out_final = torch.cat([blank.unsqueeze(-1), out], dim=-1)
        else:
            blank = ctc_output1[..., 0:1].clone()
            out = ctc_output1[..., 1:].unsqueeze(-1) + ctc_output2.unsqueeze(-2)
            out = out.permute(0, 1, 3, 2)
            out = out.reshape(out.size(0), out.size(1), -1)
            out_ = torch.cat([blank, out], dim=-1)
            out_final = self.log_softmax(out_).clone()
        
        # Helps with some numerical issues
        out_final = torch.clamp(out_final, max=0.0)
        return out_final, x_lens

    @torch.jit.ignore
    def forward_target_speaker(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        speakers = [0],
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          speakers:
            list of speakers to process one at at time for target speaker decoding.
        Returns:
          Return the output length of the audio and CTC outputs for each speaker.
        """
        assert self.factorized_token_speaker_output
        assert x_lens.ndim == 1, x_lens.shape
        nnet_output = self.encoder(x)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        assert torch.all(x_lens > 0)
        
        # Optionally downsample.
        if self.downsample:
            nnet_output = nnet_output.transpose(1, 2)
            nnet_output = nn.functional.avg_pool1d(nnet_output, kernel_size=2, stride=2)
            nnet_output = nnet_output.transpose(1, 2)
            x_lens = torch.floor((x_lens - 2) / 2 + 1).to(torch.int32)
        
        if self.use_layer_norm:
            nnet_output = self.layer_norm(nnet_output)
        
        # compute ctc log-probs
        ctc_output1 = self.ctc_output1(nnet_output)
        ctc_output2 = self.ctc_output2(nnet_output)
        out1 = self.log_softmax(ctc_output1)
        blank = out1[..., 0]
        out2 = self.log_softmax(ctc_output2)
        out = out1[..., 1:].unsqueeze(-1) + out2.unsqueeze(-2)
        out = out.permute(0, 1, 3, 2)
        outputs = []
        for spk in speakers:
            out_ = out.clone()
            speaker_mask = [i for i in range(out.size(2)) if i != spk]
            token_weight = out[:, :, speaker_mask, :].logsumexp((2, 3)).clone()
            # redistribute this weight to the remaining speakers
            blank_ = torch.logaddexp(blank, token_weight) 
            outputs.append(
                torch.clamp(
                    torch.cat([blank_.unsqueeze(-1), out_[:, :, spk, :]], dim=-1),
                    max=0.0,
                )
            )
        return x_lens, outputs
    
    @torch.jit.ignore
    def forward_tokens_only(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Tensor of dimension (N, T, C) where N is the batch size,
            T is the number of frames, and C is the feature dimension.
          x_lens:
            Tensor of dimension (N,) where N is the batch size.
          speakers:
            list of speakers to process one at at time for target speaker decoding.
        Returns:
          Return the output length of the audio and CTC outputs for each speaker.
        """
        assert self.factorized_token_speaker_output
        assert x_lens.ndim == 1, x_lens.shape
        nnet_output = self.encoder(x)[0]
        
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            x_lens = torch.floor((x_lens - width) / stride + 1)

        assert torch.all(x_lens > 0)
        
        # Optionally downsample.
        if self.downsample:
            nnet_output = nnet_output.transpose(1, 2)
            nnet_output = nn.functional.avg_pool1d(nnet_output, kernel_size=2, stride=2)
            nnet_output = nnet_output.transpose(1, 2)
            x_lens = torch.floor((x_lens - 2) / 2 + 1).to(torch.int32)
        
        if self.use_layer_norm:
            nnet_output = self.layer_norm(nnet_output)
        
        # compute ctc log-probs
        ctc_output1 = self.ctc_output1(nnet_output)
        out1 = self.log_softmax(ctc_output1)
        return out1, x_lens
    
    def freeze_encoder(self):
        try:
            for p in self.encoder.encoder.parameters():
                if p.requires_grad:
                    p.requires_grad = False
            self.frozen = True
        except AttributeError:
            for p in self.encoder.model.parameters():
                if p.requires_grad:
                    p.requires_grad = False
            self.frozen = True


    def unfreeze_encoder(self):
        try:
            for i, p in enumerate(self.encoder.encoder.parameters()):
                p.requires_grad = True
            if self.freeze_feat_extractor:
                # Freeze the feature extractor (CNN frontend)
                for param in self.encoder.feature_extractor.parameters():
                    param.requires_grad = False
            self.frozen = False
        except AttributeError:
            for i, p in enumerate(self.encoder.model.parameters()):
                p.requires_grad = True
            if self.freeze_feat_extractor:
                # Freeze the feature extractor (CNN frontend)
                for param in self.encoder.model.feature_extractor.parameters():
                    param.requires_grad = False
            self.frozen = False


class FrozenTokenTeacherSpeakerModel(nn.Module):
    """A frozen token aligner paired with a trainable speaker-only WavLM.

    The token teacher is intentionally kept outside this module's registered
    submodules.  Consequently it is excluded from the optimizer, DDP gradient
    synchronization, model averaging, and student checkpoints.  It is loaded
    afresh from ``--frozen-token-teacher-checkpoint`` when a job starts or is
    resumed.

    The legacy token-only checkpoint was trained with the first speaker block
    of :class:`MDCTCModel`.  We recover exactly that block efficiently and
    condition it on belonging to either blank or speaker block zero.  This
    avoids materializing the full token-by-speaker tensor while retaining the
    learned blank/nonblank calibration of the checkpoint.
    """

    def __init__(
        self,
        token_teacher: MDCTCModel,
        speaker_encoder: nn.Module,
        encoder_dim: int,
        vocab_size: int,
        num_speakers: int,
        freeze_feat_extractor: bool = True,
        layer_norm: bool = False,
        downsample: bool = True,
    ):
        super().__init__()
        for parameter in token_teacher.parameters():
            parameter.requires_grad = False
        token_teacher.eval()
        # Bypass nn.Module.__setattr__: the immutable teacher must not enter
        # state_dict(), parameters(), DDP, or the averaged student model.
        object.__setattr__(self, "_token_teacher", token_teacher)

        try:
            for parameter in speaker_encoder.feature_extractor.parameters():
                parameter.requires_grad = False
        except AttributeError:
            for parameter in speaker_encoder.model.feature_extractor.parameters():
                parameter.requires_grad = False

        self.encoder = speaker_encoder
        self.ctc_output2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, num_speakers),
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.vocab_size = vocab_size
        self.freeze_feat_extractor = freeze_feat_extractor
        self.use_layer_norm = layer_norm
        self.downsample = downsample
        self.factorized_token_speaker_output = True
        self.frozen = False

    @property
    def token_teacher(self) -> Optional[MDCTCModel]:
        return self._token_teacher

    def release_token_teacher(self) -> Optional[MDCTCModel]:
        """Temporarily detach the teacher, e.g. while deepcopying model_avg."""
        teacher = self._token_teacher
        object.__setattr__(self, "_token_teacher", None)
        return teacher

    def attach_token_teacher(self, teacher: MDCTCModel) -> None:
        if teacher is None:
            raise ValueError("Cannot attach an empty token teacher")
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
        object.__setattr__(self, "_token_teacher", teacher)

    def move_token_teacher(self, device: torch.device) -> None:
        if self._token_teacher is None:
            raise RuntimeError("The frozen token teacher is not attached")
        self._token_teacher.to(device)
        self._token_teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._token_teacher is not None:
            # The teacher's dropout must never be enabled by model.train().
            self._token_teacher.eval()
        return self

    @staticmethod
    def _output_lengths(x_lens: torch.Tensor, downsample: bool) -> torch.Tensor:
        for width, stride in [
            (10, 5),
            (3, 2),
            (3, 2),
            (3, 2),
            (3, 2),
            (2, 2),
            (2, 2),
        ]:
            x_lens = torch.floor((x_lens - width) / stride + 1)
        if downsample:
            x_lens = torch.floor((x_lens - 2) / 2 + 1)
        return x_lens.to(torch.int32)

    @staticmethod
    def _postprocess_encoder_output(
        output: torch.Tensor,
        downsample: bool,
        use_layer_norm: bool,
        layer_norm: nn.Module,
    ) -> torch.Tensor:
        if downsample:
            output = nn.functional.avg_pool1d(
                output.transpose(1, 2), kernel_size=2, stride=2
            ).transpose(1, 2)
        if use_layer_norm:
            output = layer_norm(output)
        return output

    def _teacher_token_scores(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        teacher = self._token_teacher
        if teacher is None:
            raise RuntimeError("The frozen token teacher is not attached")
        with torch.no_grad():
            hidden = teacher.encoder(x)[0]
            teacher_lens = self._output_lengths(x_lens, teacher.downsample)
            hidden = self._postprocess_encoder_output(
                hidden,
                teacher.downsample,
                teacher.use_layer_norm,
                teacher.layer_norm,
            )
            token_logits = teacher.ctc_output1(hidden)
            if token_logits.size(-1) != self.vocab_size:
                raise RuntimeError(
                    "Teacher vocabulary mismatch: "
                    f"{token_logits.size(-1)} != {self.vocab_size}"
                )

            if teacher.factorized_token_speaker_output:
                token_scores = teacher.log_softmax(token_logits)
            else:
                # These are exactly the logits for blank and the first speaker
                # block used by the legacy token-only graph, conditioned on
                # that block rather than allocating probability to the seven
                # unused speaker replicas.
                speaker_zero = teacher.ctc_output2(hidden)[..., :1]
                conditioned_logits = torch.cat(
                    [
                        token_logits[..., :1],
                        token_logits[..., 1:] + speaker_zero,
                    ],
                    dim=-1,
                )
                token_scores = teacher.log_softmax(conditioned_logits)
        return token_scores.detach(), teacher_lens

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        speaker_weight: float = 1.0,
        return_factors: bool = False,
    ):
        assert x_lens.ndim == 1, x_lens.shape
        token_scores, teacher_lens = self._teacher_token_scores(x, x_lens)

        hidden = self.encoder(x)[0]
        speaker_lens = self._output_lengths(x_lens, self.downsample)
        hidden = self._postprocess_encoder_output(
            hidden, self.downsample, self.use_layer_norm, self.layer_norm
        )
        speaker_scores = speaker_weight * self.log_softmax(self.ctc_output2(hidden))

        if token_scores.shape[:2] != speaker_scores.shape[:2] or not torch.equal(
            teacher_lens, speaker_lens
        ):
            raise RuntimeError(
                "Frozen token teacher and speaker student frame grids differ: "
                f"token={tuple(token_scores.shape)} speaker={tuple(speaker_scores.shape)} "
                f"token_lens={teacher_lens.tolist()} "
                f"speaker_lens={speaker_lens.tolist()}"
            )
        if return_factors:
            return token_scores, speaker_scores, speaker_lens

        blank = token_scores[..., :1]
        joint = token_scores[..., 1:].unsqueeze(-2) + speaker_scores.unsqueeze(-1)
        joint = joint.reshape(joint.size(0), joint.size(1), -1)
        return torch.cat([blank, joint], dim=-1).clamp_max(0.0), speaker_lens

    def freeze_encoder(self) -> None:
        try:
            parameters = self.encoder.encoder.parameters()
        except AttributeError:
            parameters = self.encoder.model.parameters()
        for parameter in parameters:
            parameter.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        try:
            for parameter in self.encoder.encoder.parameters():
                parameter.requires_grad = True
            if self.freeze_feat_extractor:
                for parameter in self.encoder.feature_extractor.parameters():
                    parameter.requires_grad = False
        except AttributeError:
            for parameter in self.encoder.model.parameters():
                parameter.requires_grad = True
            if self.freeze_feat_extractor:
                for parameter in self.encoder.model.feature_extractor.parameters():
                    parameter.requires_grad = False
        self.frozen = False


class SpeakerDiarizationModel(nn.Module):
    """WavLM encoder with a diarization-only frame classifier.

    Unlike :class:`MDCTCModel` and :class:`FrozenTokenTeacherSpeakerModel`,
    this model has no token head or token teacher.  ``output_dim`` is either
    the number of FIFO speaker slots (independent BCE) or the number of
    admissible powerset classes (cross entropy).
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        output_dim: int,
        freeze_feat_extractor: bool = True,
        layer_norm: bool = False,
        downsample: bool = True,
    ):
        super().__init__()
        try:
            feature_extractor = encoder.feature_extractor
        except AttributeError:
            feature_extractor = encoder.model.feature_extractor
        if freeze_feat_extractor:
            for parameter in feature_extractor.parameters():
                parameter.requires_grad = False

        self.encoder = encoder
        self.output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, output_dim),
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.freeze_feat_extractor = freeze_feat_extractor
        self.use_layer_norm = layer_norm
        self.downsample = downsample
        self.frozen = False

    @staticmethod
    def _output_lengths(x_lens: torch.Tensor, downsample: bool) -> torch.Tensor:
        for width, stride in [
            (10, 5),
            (3, 2),
            (3, 2),
            (3, 2),
            (3, 2),
            (2, 2),
            (2, 2),
        ]:
            x_lens = torch.floor((x_lens - width) / stride + 1)
        if downsample:
            x_lens = torch.floor((x_lens - 2) / 2 + 1)
        return x_lens.to(torch.int32)

    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        **_unused,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x_lens.ndim == 1, x_lens.shape
        hidden = self.encoder(x)[0]
        output_lens = self._output_lengths(x_lens, self.downsample)
        if self.downsample:
            hidden = nn.functional.avg_pool1d(
                hidden.transpose(1, 2), kernel_size=2, stride=2
            ).transpose(1, 2)
        if self.use_layer_norm:
            hidden = self.layer_norm(hidden)
        return self.output(hidden), output_lens

    def freeze_encoder(self) -> None:
        try:
            parameters = self.encoder.encoder.parameters()
        except AttributeError:
            parameters = self.encoder.model.parameters()
        for parameter in parameters:
            parameter.requires_grad = False
        self.frozen = True

    def unfreeze_encoder(self) -> None:
        try:
            for parameter in self.encoder.encoder.parameters():
                parameter.requires_grad = True
            if self.freeze_feat_extractor:
                for parameter in self.encoder.feature_extractor.parameters():
                    parameter.requires_grad = False
        except AttributeError:
            for parameter in self.encoder.model.parameters():
                parameter.requires_grad = True
            if self.freeze_feat_extractor:
                for parameter in self.encoder.model.feature_extractor.parameters():
                    parameter.requires_grad = False
        self.frozen = False
