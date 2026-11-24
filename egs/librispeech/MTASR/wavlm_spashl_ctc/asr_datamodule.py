# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import gzip
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    SupervisionSegment,
    fastcopy,
    load_manifest,
    load_manifest_lazy,
)
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from dataset import K2MultiTalkerSpeechRecognitionDataset
from mix_to_mono_dataset import K2MixToMonoDataset
from vendor.lhotse.cut_splice3 import (
    ApproximateResumeDynamicBucketingSampler,
    CutSpliceIterable,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from icefall.utils import str2bool
from text_normalization import normalize_english_transcript


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class _SkipSamplerBatches:
    """Replay a deterministic sampler while discarding an initial prefix."""

    def __init__(self, sampler, num_batches: int):
        self.sampler = sampler
        self.num_batches = num_batches

    def __iter__(self):
        iterator = iter(self.sampler)
        logging.info("Replaying and skipping %d sampler batches", self.num_batches)
        for index in range(self.num_batches):
            next(iterator)
            if (index + 1) % 50000 == 0:
                logging.info("Skipped %d/%d sampler batches", index + 1, self.num_batches)
        logging.info("Finished deterministic sampler replay")
        yield from iterator

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        return self.sampler.load_state_dict(state_dict)

    def set_epoch(self, epoch):
        return self.sampler.set_epoch(epoch)


class _RepeatCutSources:
    """Independently repeat synthetic and YODAS streams, then mix by type."""

    def __init__(
        self,
        manifest_paths,
        shar_dirs,
        yodas_schemes,
        seed: int,
        yodas_mix_weight: float,
    ):
        self.manifest_paths = tuple(Path(p) for p in manifest_paths)
        self.shar_dirs = tuple(Path(p) for p in shar_dirs)
        self.yodas_schemes = tuple(yodas_schemes)
        self.seed = seed
        self.yodas_mix_weight = yodas_mix_weight

    def _repeat_standard_sources(self):
        epoch = 0
        while True:
            cutsets = [load_manifest_lazy(path) for path in self.manifest_paths]
            cutsets.extend(CutSet.from_shar(in_dir=path) for path in self.shar_dirs)
            if len(cutsets) == 1:
                yield from cutsets[0]
            else:
                yield from CutSet.mux(*cutsets, seed=self.seed + epoch)
            epoch += 1

    def _repeat_yodas_sources(self):
        epoch = 0
        while True:
            cutsets = [CutSet(_YodasSharGroups(**spec)) for spec in self.yodas_schemes]
            if len(cutsets) == 1:
                yield from cutsets[0]
            else:
                yield from CutSet.mux(*cutsets, seed=self.seed + epoch)
            epoch += 1

    def __iter__(self):
        has_standard = bool(self.manifest_paths or self.shar_dirs)
        has_yodas = bool(self.yodas_schemes)
        if has_standard and has_yodas:
            # Both inputs are infinite repeat streams, so the requested corpus
            # ratio remains stable rather than changing when YODAS exhausts.
            yield from CutSet.mux(
                CutSet(self._repeat_standard_sources()),
                CutSet(self._repeat_yodas_sources()),
                weights=[1.0 - self.yodas_mix_weight, self.yodas_mix_weight],
                seed=self.seed,
            )
        elif has_standard:
            yield from self._repeat_standard_sources()
        elif has_yodas:
            yield from self._repeat_yodas_sources()


class _YodasSharGroups:
    """Metadata-only 60 s views over selected parent recordings in YODAS SHAR."""

    def __init__(
        self,
        scheme_dir,
        hypothesis="doverlap",
        max_pause=1.0,
        max_duration=60.0,
        min_speakers=2,
        max_speakers=8,
    ):
        self.scheme_dir = Path(scheme_dir)
        self.hypothesis = hypothesis
        self.max_pause = max_pause
        self.max_duration = max_duration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    @staticmethod
    def _swap_supervisions(cut, hypothesis):
        h = cut.custom["hypotheses"][hypothesis]
        sups = [
            SupervisionSegment(
                id=f"{cut.id}-sup{i:03d}",
                recording_id=cut.recording_id,
                start=round(seg["start"], 3),
                duration=round(seg["end"] - seg["start"], 3),
                channel=0,
                text=normalize_english_transcript(seg["text"]),
                language=h.get("language"),
                speaker=seg["speaker"],
            )
            for i, seg in enumerate(h["segments"])
            if round(seg["end"] - seg["start"], 3) > 0
            and normalize_english_transcript(seg["text"])
        ]
        return fastcopy(cut, supervisions=sups)

    def _split(self, cut):
        sups = sorted(cut.supervisions, key=lambda s: s.start)
        if not sups:
            return
        groups, group, end = [], [sups[0]], sups[0].end
        for sup in sups[1:]:
            if sup.start - end <= self.max_pause:
                group.append(sup)
                end = max(end, sup.end)
            else:
                groups.append(group)
                group, end = [sup], sup.end
        groups.append(group)
        for group_idx, group in enumerate(groups):
            start, end = group[0].start, max(s.end for s in group)
            piece_idx = 0
            while start < end:
                piece_end = min(start + self.max_duration, end)
                piece = cut.truncate(
                    offset=start,
                    duration=piece_end - start,
                    keep_excessive_supervisions=False,
                    preserve_id=True,
                ).with_id(f"{cut.id}-g{group_idx}-{piece_idx}")
                num_speakers = len({s.speaker for s in piece.supervisions})
                if self.min_speakers <= num_speakers <= self.max_speakers:
                    yield piece
                start = piece_end
                piece_idx += 1

    def __iter__(self):
        meta_dir = self.scheme_dir.parent / f"{self.scheme_dir.name}.meta"
        keep_ids = set((meta_dir / "keep_ids.txt").read_text().splitlines())
        scheme = json.loads((meta_dir / "scheme.json").read_text())
        shar_dir = Path(scheme["shar_dir"])

        cuts_shards, recording_shards = [], []
        shard_names = scheme.get("kept_source_shards")
        candidate_paths = (
            [shar_dir / name for name in shard_names]
            if shard_names
            else sorted(shar_dir.glob("cuts.*.jsonl.gz"))
        )
        for cuts_path in candidate_paths:
            if not shard_names:
                with gzip.open(cuts_path, "rt", encoding="utf-8") as stream:
                    if not any(json.loads(line)["id"] in keep_ids for line in stream):
                        continue
            suffix = cuts_path.name.removeprefix("cuts.").removesuffix(".jsonl.gz")
            cuts_shards.append(cuts_path)
            recording_shards.append(shar_dir / f"recording.{suffix}.tar")

        parents = CutSet.from_shar(
            fields={"cuts": cuts_shards, "recording": recording_shards}
        )
        for parent in parents:
            if parent.id not in keep_ids:
                continue
            parent = self._swap_supervisions(parent, self.hypothesis)
            yield from self._split(parent)


class LibriSpeechAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).
    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction
    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=100.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--max-cuts",
            type=int,
            default=None,
            help="Optional hard limit on the number of cuts in each batch.",
        )
        group.add_argument(
            "--max-train-cut-duration",
            type=float,
            default=60.0,
            help=(
                "Reject individual training cuts longer than this many seconds. "
                "This is independent of --max-duration, which controls the "
                "pooled batch cost."
            ),
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=100,
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--num-cuts-for-bins-estimate",
            type=int,
            default=10000,
            help=(
                "Number of cuts sampled to estimate dynamic bucket boundaries. "
                "Use a small value such as 100 for smoke tests."
            ),
        )
        group.add_argument(
            "--sampler-buffer-size",
            type=int,
            default=20000,
            help=(
                "Number of cuts used to pre-populate dynamic buckets before "
                "sampling. Use a small value such as 200 for smoke tests."
            ),
        )
        group.add_argument(
            "--fixed-sampler-skip-batches",
            type=int,
            default=0,
            help=(
                "For deterministic fixed-manifest reproduction, reconstruct "
                "the original sampler stream and discard this many batches "
                "before loading audio."
            ),
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=5,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--normalize-loudness",
            type=str2bool,
            default=False,
            help="Normalize the loudness of each cutsplice segment",
        )
        group.add_argument(
            "--max-splice-duration",
            type=float,
            default=30,
        )
        group.add_argument(
            "--max-splices",
            type=int,
            default=4,
        )
        group.add_argument(
            "--min-splices",
            type=int,
            default=2,
        )
        group.add_argument(
            "--max-unique",
            type=int,
            default=3,
        )
        group.add_argument(
            "--max-num-overlaps",
            type=int,
            default=3,
        )
        group.add_argument(
            "--overlap",
            type=float,
            default=0.6,
        )
        group.add_argument(
            "--min-overlap",
            type=float,
            default=0.5,
        )
        group.add_argument(
            "--drift",
            type=float,
            default=1.0,
        )
        group.add_argument(
            "--reverb",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--max-snr",
            type=float,
            default=30.0,
        )
        group.add_argument(
            "--splice-schedule-increment",
            type=float,
            default=4e-05,
        )
        group.add_argument(
            "--duration-increment",
            type=float,
            default=1e-03,
        )
        group.add_argument(
            "--allow-self-overlap",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--train-cut-manifests",
            nargs="+",
            default=None,
            help=(
                "One or more pre-built Lhotse CutSet manifests to use directly "
                "for training. This bypasses on-the-fly CutSpliceIterable "
                "simulation and repeatedly multiplexes the supplied manifests."
            ),
        )
        group.add_argument(
            "--train-shar-dirs",
            nargs="+",
            default=None,
            help=(
                "One or more finite Lhotse SHAR directories to mix with "
                "--train-cut-manifests and repeat for update-based training."
            ),
        )
        group.add_argument(
            "--train-yodas-scheme-dirs",
            nargs="+",
            default=None,
            help=(
                "Filtered YODAS scheme directories created by "
                "build_filtered_cutset.py. Sixty-second supervision-only "
                "views are generated while streaming the original SHAR."
            ),
        )
        group.add_argument("--yodas-hypothesis", default="doverlap")
        group.add_argument("--yodas-group-max-pause", type=float, default=1.0)
        group.add_argument("--yodas-group-max-duration", type=float, default=60.0)
        group.add_argument("--yodas-min-speakers", type=int, default=2)
        group.add_argument("--yodas-max-speakers", type=int, default=8)
        group.add_argument(
            "--yodas-mix-weight",
            type=float,
            default=0.5,
            help=(
                "Probability of drawing from the independently repeated YODAS "
                "stream when mixing it with fixed manifests (default: 0.5)."
            ),
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        input_transforms = []
        logging.info("About to create train dataset")
        train = K2MultiTalkerSpeechRecognitionDataset(
            cut_transforms=[],
            input_transforms=input_transforms,
            sort_strategy=self.args.sort_strategy,
        )

        fixed_training_cuts = bool(
            self.args.train_cut_manifests
            or self.args.train_shar_dirs
            or self.args.train_yodas_scheme_dirs
        )
        sampler_cls = (
            DynamicBucketingSampler
            if fixed_training_cuts
            else ApproximateResumeDynamicBucketingSampler
        )
        logging.info("Using %s.", sampler_cls.__name__)
        train_sampler = sampler_cls(
            cuts_train,
            max_duration=self.args.max_duration,
            max_cuts=self.args.max_cuts,
            shuffle=False,
            num_buckets=self.args.num_buckets,
            num_cuts_for_bins_estimate=self.args.num_cuts_for_bins_estimate,
            buffer_size=self.args.sampler_buffer_size,
            drop_last=True,
            quadratic_duration=20,
        )
            #buffer_size=100,
            #num_cuts_for_bins_estimate=100,
            #duration_bins=list(range(10, 60, 2)),

        logging.info("About to create train dataloader")
        
        if sampler_state_dict is not None and not fixed_training_cuts:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)
        elif sampler_state_dict is not None:
            logging.info(
                "Starting the fixed-cut sampler fresh on resume; model and "
                "optimizer state are restored, but fixed cuts do not have a "
                "CutSplice curriculum to reconstruct."
            )

        if self.args.fixed_sampler_skip_batches > 0:
            if not fixed_training_cuts:
                raise ValueError(
                    "--fixed-sampler-skip-batches requires fixed training sources"
                )
            train_sampler = _SkipSamplerBatches(
                train_sampler, self.args.fixed_sampler_skip_batches
            )

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        validate = K2MultiTalkerSpeechRecognitionDataset(
            return_cuts = self.args.return_cuts,
            cut_transforms=[],
            sort_strategy=self.args.sort_strategy,
            downmix_multichannel=getattr(self.args, "downmix_multichannel", False),
            audio_sampling_rate=getattr(self.args, "input_sampling_rate", 0),
        )
        
        # DynamicBucketingSampler requires num_buckets > 1.  A single bucket
        # means bucketing is disabled, so use the ordinary sampler instead.
        # Also retain the cheap path for an eager one-cut diagnostic set.
        if self.args.num_buckets <= 1 or (
            not cuts_valid.is_lazy and len(cuts_valid) == 1
        ):
            valid_sampler = SimpleCutSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                max_cuts=self.args.max_cuts,
                shuffle=False,
            )
        else:
            valid_sampler = DynamicBucketingSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
                num_buckets=self.args.num_buckets,
                num_cuts_for_bins_estimate=self.args.num_cuts_for_bins_estimate,
                buffer_size=self.args.sampler_buffer_size,
            )
        
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2MultiTalkerSpeechRecognitionDataset(
            return_cuts=self.args.return_cuts,
            sort_strategy=self.args.sort_strategy,
            downmix_multichannel=getattr(self.args, "downmix_multichannel", False),
            audio_sampling_rate=getattr(self.args, "input_sampling_rate", 0),
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=False
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    def mix_to_mono_dataloader(self, cuts: CutSet) -> DataLoader:
        logging.info("About to create mix_to_mono dataloader")
        test = K2MixToMonoDataset(
            return_cuts=self.args.return_cuts,
            sort_strategy=self.args.sort_strategy,
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=False
        )
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl


    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        if (
            self.args.train_cut_manifests
            or self.args.train_shar_dirs
            or self.args.train_yodas_scheme_dirs
        ):
            paths = [Path(p) for p in (self.args.train_cut_manifests or [])]
            shar_dirs = [Path(p) for p in (self.args.train_shar_dirs or [])]
            yodas_dirs = [
                Path(p) for p in (self.args.train_yodas_scheme_dirs or [])
            ]
            missing = [str(p) for p in paths if not p.is_file()]
            missing.extend(str(p) for p in shar_dirs if not p.is_dir())
            missing.extend(str(p) for p in yodas_dirs if not p.is_dir())
            if missing:
                raise FileNotFoundError(
                    "Missing fixed training CutSet manifest(s): " + ", ".join(missing)
                )
            logging.info(
                "Using %d fixed manifests, %d SHAR dirs, and %d YODAS schemes: %s",
                len(paths),
                len(shar_dirs),
                len(yodas_dirs),
                ", ".join(map(str, paths + shar_dirs + yodas_dirs)),
            )
            # Training is update-based and normally consumes an infinite
            # CutSplice stream. Consume every finite FastMSS shard once per
            # pass, then begin another deterministically re-interleaved pass.
            yodas_schemes = [
                {
                    "scheme_dir": path,
                    "hypothesis": self.args.yodas_hypothesis,
                    "max_pause": self.args.yodas_group_max_pause,
                    "max_duration": self.args.yodas_group_max_duration,
                    "min_speakers": self.args.yodas_min_speakers,
                    "max_speakers": self.args.yodas_max_speakers,
                }
                for path in yodas_dirs
            ]
            if yodas_schemes and paths + shar_dirs and not (
                0.0 < self.args.yodas_mix_weight < 1.0
            ):
                raise ValueError(
                    "--yodas-mix-weight must be between 0 and 1 when mixing "
                    f"corpora; got {self.args.yodas_mix_weight}"
                )
            return CutSet(
                _RepeatCutSources(
                    paths,
                    shar_dirs,
                    yodas_schemes,
                    seed=self.args.seed,
                    yodas_mix_weight=self.args.yodas_mix_weight,
                )
            )
        speakers = list(Path("data/manifests/librispeech_speakers").rglob("*.jsonl.gz"))
        num_spks = len(speakers) 
        weight = 1/num_spks
        sim_weight = 0.0
        cut_info = []
        for s in speakers:
            cut_info.append((s.stem.split("_")[-1].split(".")[0], (1-sim_weight)*weight, s))
        
        cutsets, weights, names = [], [], []
        for n, w, p in cut_info: 
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=20,
            final_max_duration=self.args.max_splice_duration,
            max_duration_increment=self.args.duration_increment,
            max_num_overlaps=self.args.max_num_overlaps,
            max_splices=1,
            min_splices=1,
            final_max_splices=self.args.max_splices,
            final_min_splices=self.args.min_splices,
            max_splices_schedule_increment=self.args.splice_schedule_increment,
            min_splices_schedule_increment=self.args.splice_schedule_increment,
            max_unique=self.args.max_unique,
            min_unique=self.args.max_unique,
            max_snr=[2*self.args.max_snr * (random.random() - 0.5) for i in range(num_spks)],
            normalize_loudness=self.args.normalize_loudness,
            reverb=self.args.reverb,
            sampling_rate=16000,
            drift=self.args.drift,
            overlap=self.args.overlap,
            min_overlap=self.args.min_overlap,
            self_overlap=self.args.allow_self_overlap,
        )

        return CutSet(cs_iter)

    @lru_cache()
    def train_ami_cuts(self) -> CutSet:
        recordings = list(Path("data/manifests/ami_recordings").rglob("*.jsonl.gz"))
        num_recos = len(recordings)
        weight = 1/num_recos
        cut_info = []
        for s in recordings:
            cut_info.append((s.stem.split("_")[-1].split(".")[0], weight, s)) 

        cutsets, weights, names = [], [], []
        for n, w, p in cut_info:
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=20,
            final_max_duration=self.args.max_splice_duration,
            max_duration_increment=self.args.duration_increment,
            max_num_overlaps=self.args.max_num_overlaps,
            max_splices=1,
            min_splices=1,
            final_max_splices=self.args.max_splices,
            final_min_splices=self.args.min_splices,
            max_splices_schedule_increment=self.args.splice_schedule_increment,
            min_splices_schedule_increment=self.args.splice_schedule_increment,
            max_unique=self.args.max_unique,
            min_unique=self.args.max_unique,
            max_snr=[2*self.args.max_snr * (random.random() - 0.5) for i in range(num_recos)], 
            normalize_loudness=self.args.normalize_loudness,
            reverb=self.args.reverb,
            sampling_rate=16000,
            drift=self.args.drift,
            overlap=self.args.overlap,
            min_overlap=self.args.min_overlap,
            self_overlap=self.args.allow_self_overlap,
        )

        return CutSet(cs_iter)

    @lru_cache()
    def libricss_cuts(self) -> CutSet:
        logging.info("About to get libricss cuts")
        cuts = load_manifest_lazy("data/manifests/libricss-ihm-mix_segments_all.jsonl.gz")
        cuts = cuts.filter(lambda c: c.id.split("_")[1].split("-")[0] != "session0")
        return cuts

    @lru_cache()
    def single_speaker_cuts(self) -> CutSet:
        test_other = load_manifest_lazy("./data/manifests/cuts_librispeech_test-other.jsonl.gz")
        test_clean = load_manifest_lazy("./data/manifests/cuts_librispeech_test-clean.jsonl.gz")
        return test_other, test_clean

    @lru_cache()
    def libri2mix_dev_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_clean_sc_dev_cutset.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def libri2mix_test_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_clean_sc_test_cutset.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def libri2mix_test_both_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_both_sc_test_cutset.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def libri3mix_test_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri3mix_mix_clean_sc_test_cutset.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def librispeechmix_2_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/cuts_librispeechmix_test-clean-2mix.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def librispeechmix_3_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/cuts_librispeechmix_test-clean-3mix.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def valid_cuts(self, no_overlap=False) -> CutSet:
        logging.info("About to get dev cuts")
        if no_overlap:
            cuts = load_manifest_lazy("./data/manifests/cuts_librispeech_dev-clean.jsonl.gz")
            cuts_ = []
            total = 0
            for c in cuts:
                if total > 2*3600:
                    break
                cuts_.append(c)
                total += c.duration
            return CutSet(cuts_)

        return load_manifest_lazy("data/manifests/cuts_librispeech_dev_synth.jsonl.gz")        
        cut_info = [
            ('dev-clean', 0.5, "./data/manifests/cuts_librispeech_dev-clean.jsonl.gz"),
            ('dev-other', 0.5, "./data/manifests/cuts_librispeech_dev-other.jsonl.gz"),
        ]
        cutsets, weights, names = [], [], []
        for n, w, p in cut_info: 
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=30,
            max_splices=2,
            min_splices=2,
            max_snr=[0, 0],
            final_max_splices=2,
            max_unique=2,
            normalize_loudness=False,
            sampling_rate=16000,
        )
        
        # Just do two hours of data
        total_duration = 0
        cuts = []
        for c in cs_iter:
            if total_duration > 2*3600:
                break
            total_duration += c.duration
            cuts.append(c)     
        return CutSet(cuts)


    @lru_cache()
    def valid_ami_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        cuts = load_manifest_lazy("data/manifests/cuts_ami_dev_pause0.5.jsonl.gz")
        cuts = cuts.filter(lambda c: c.duration <= 40)
        cuts_ = []
        total_duration = 0
        for c in cuts:
            if total_duration > 2*3600:
                break
            total_duration += c.duration
            cuts_.append(c)
        return CutSet(cuts_)
        #if self.args.normalize_loudness:
        #    return cuts.normalize_loudness(-23)

        recordings = list(Path("data/manifests/ami_recordings_dev").rglob("*.jsonl.gz"))
        num_recos = len(recordings)
        weight = 1/num_recos
        cut_info = []
        for s in recordings:
            cut_info.append((s.stem.split("_")[-1].split(".")[0], weight, s))
        
        cutsets, weights, names = [], [], []
        for n, w, p in cut_info: 
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=30,
            max_splices=2,
            min_splices=2,
            max_snr=[0, 0],
            final_max_splices=2,
            max_unique=2,
            normalize_loudness=False,
            sampling_rate=16000,
        )
        
        # Just do two hours of data
        total_duration = 0
        cuts = []
        for c in cs_iter:
            if total_duration > 2*3600:
                break
            total_duration += c.duration
            cuts.append(c)     
        return CutSet(cuts)

    @lru_cache()
    def synth_cuts(self) -> CutSet:
        logging.info("About to get snythetic cuts")
        cuts = load_manifest_lazy("data/manifests/cuts_librispeech_dev_synth.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def synth2_cuts(self, i, ds="dev") -> CutSet:
        logging.info("About to get snythetic cuts")
        cuts = load_manifest_lazy(f"data/manifests/cuts_librispeech_{ds}_synth_{i}spk_8splices.jsonl.gz")
        if self.args.normalize_loudness:
            return cuts.normalize_loudness(-23)
        return cuts

    
    @lru_cache()
    def aed_cuts(self) -> CutSet:
        logging.info("About to get audioset cuts")
        return load_manifest_lazy("data/manifests/audioset_eval_whole/cuts_0_mapped.jsonl.gz").subset(first=100) 

    @lru_cache()
    def ami_dev(self) -> CutSet:
        logging.info("About to get ami dev cuts")
        cuts = load_manifest_lazy("data/manifests/cuts_ami_dev_pause0.5.jsonl.gz")
        #cuts = cuts.filter(lambda c: c.duration <= 40)
        #if self.args.normalize_loudness:
        #    return cuts.normalize_loudness(-23)
        return cuts

    @lru_cache()
    def ami_test(self) -> CutSet:
        logging.info("About to get ami test cuts")
        return load_manifest_lazy("data/manifests/cuts_ami_test_pause0.5.jsonl.gz")


def test():
    parser = argparse.ArgumentParser()
    Scale23SpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    adm = Scale23SpeechAsrDataModule(args)

    cuts = adm.train_cuts()
    dl = adm.train_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break

    cuts = adm.dev_cuts()
    dl = adm.valid_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break


if __name__ == "__main__":
    test()
