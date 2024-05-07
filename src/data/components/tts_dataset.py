import os
import random
from typing import Optional

import torch
import torch.utils.data

from src.data.components.mel_processing import (mel_spectrogram_torch, spectrogram_torch)
from src.data.components.text import cleaned_text_to_sequence, text_to_sequence
from src.data.components.utils import intersperse, load_filepaths_and_text, load_wav_to_torch


"""Multi speaker version"""
class TextAudioSpeakerDataset(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, 
            audiopaths_sid_text: str, 
            use_mel_posterior_encoder: bool = True,
            cleaned_text: bool = True,
            text_cleaners: list[str] = ["english_cleaners2"],
            min_text_len: int = 1,
            max_text_len: int = 200,
            max_wav_value: float = 32768.0,
            min_audio_len: int = 4096,
            sampling_rate: int = 16000,
            filter_length: int = 512,
            hop_length: int = 128,
            win_length: int = 512, # win_length / sampling_rate ~= 0.032s
            n_mel_channels: int = 80,
            mel_fmin: float = 0.0,
            mel_fmax: Optional[float] = None,
            add_blank: bool = False,
            n_speakers: int = 3000,
            use_ext_spk_emb: bool = False,
            random_seed: int = 42,
            test: bool = False,
        ):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = text_cleaners
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.use_mel_spec_posterior = use_mel_posterior_encoder
        self.use_ext_spk_emb = use_ext_spk_emb
        self.n_mel_channels = n_mel_channels
        self.cleaned_text = cleaned_text
        self.add_blank = add_blank
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.min_audio_len = min_audio_len
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_speakers = n_speakers

        print(f'max_text_len : {self.max_text_len}')
        print(f'min_audio_len : {self.min_audio_len}')

        if not test:
            random.seed(random_seed)
            random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if not os.path.isfile(audiopath):
                print(f"{audiopath} is not file)")
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                length = os.path.getsize(audiopath) // (2 * self.hop_length) # we use 16bit audio data
                if length < self.min_audio_len // self.hop_length:
                    print(f"{audiopath} less then min_audio_len")
                    continue
                lengths.append(length)
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(
            'dataset length:', len(self.lengths)
        )  # if we use large corpus dataset, we can check how much time it takes.

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = (
            audiopath_sid_text[0],
            audiopath_sid_text[1],
            audiopath_sid_text[2],
        )
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        if self.use_ext_spk_emb:
            embpath = audiopath.replace(".wav", ".pth")
            sid = self.get_sid(sid, embpath)
        else:
            sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio.unsqueeze(0) # audio is already normalized
        spec_filename = filename.replace(".wav", f".spec_{self.hop_length}.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(f".spec_{self.hop_length}.pt", f".mel_{self.hop_length}.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.mel_fmin,
                    self.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text, self.text_cleaners)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid, filename=None):
        if filename is not None:
            sid = torch.FloatTensor(torch.load(filename))
        else:
            sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, hparams):
        self.use_ext_spk_emb = getattr(hparams, "use_ext_spk_emb", False)
        self.return_ids = getattr(hparams, "return_ids", False)

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if not self.use_ext_spk_emb:
            sid = torch.LongTensor(len(batch))
        else:
            sid = torch.FloatTensor(len(batch), 256)

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return {
            "text": text_padded,
            "text_len": text_lengths,
            "spec": spec_padded,
            "spec_len": spec_lengths,
            "wav": wav_padded,
            "wav_len": wav_lengths,
            "sid": sid,
            "ids_sorted_decreasing": ids_sorted_decreasing,
        }
        return {
            "text": text_padded,
            "text_len": text_lengths,
            "spec": spec_padded,
            "spec_len": spec_lengths,
            "wav": wav_padded,
            "wav_len": wav_lengths,
            "sid": sid,
        }


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

        print(f'num samples per bucket: {self.num_samples_per_bucket}')
        print(f'total size: {self.total_size}')

    def _create_buckets(self):
        from collections import defaultdict
        removed_buckets = defaultdict(int)

        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
            else:
                removed_buckets[(length // 100) * 100] += 1

        for k, v in removed_buckets.items():
            print(f'bucket length {k} ~ {k + 100} : {v} samples removed')

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
                
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size



# class TextAudioLoader(torch.utils.data.Dataset):
#     """
#     1) loads audio, text pairs
#     2) normalizes text and converts them to sequences of integers
#     3) computes spectrograms from audio files.
#     """

#     def __init__(self, 
#             audiopaths_and_text, 
#             hparams
#         ):
#         self.hparams = hparams
#         self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
#         self.text_cleaners = hparams.text_cleaners
#         self.max_wav_value = hparams.max_wav_value
#         self.sampling_rate = hparams.sampling_rate
#         self.filter_length = hparams.filter_length
#         self.hop_length = hparams.hop_length
#         self.win_length = hparams.win_length
#         self.sampling_rate = hparams.sampling_rate

#         self.use_mel_spec_posterior = getattr(
#             hparams, "use_mel_posterior_encoder", False
#         )
#         if self.use_mel_spec_posterior:
#             self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
#         self.cleaned_text = getattr(hparams, "cleaned_text", False)

#         self.add_blank = hparams.add_blank
#         self.min_text_len = getattr(hparams, "min_text_len", 1)
#         self.max_text_len = getattr(hparams, "max_text_len", 190)

#         random.seed(1234)
#         random.shuffle(self.audiopaths_and_text)
#         self._filter()

#     def _filter(self):
#         """
#         Filter text & store spec lengths
#         """
#         # Store spectrogram lengths for Bucketing
#         # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
#         # spec_length = wav_length // hop_length

#         audiopaths_and_text_new = []
#         lengths = []
#         for audiopath, text in self.audiopaths_and_text:
#             if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
#                 audiopaths_and_text_new.append([audiopath, text])
#                 lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
#         self.audiopaths_and_text = audiopaths_and_text_new
#         self.lengths = lengths

#     def get_audio_text_pair(self, audiopath_and_text):
#         # separate filename and text
#         audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
#         text = self.get_text(text)
#         spec, wav = self.get_audio(audiopath)
#         return (text, spec, wav)

#     def get_audio(self, filename):
#         # TODO : if linear spec exists convert to mel from existing linear spec
#         audio, sampling_rate = load_wav_to_torch(filename)
#         if sampling_rate != self.sampling_rate:
#             raise ValueError(
#                 f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
#             )
#         audio_norm = audio / self.max_wav_value
#         audio_norm = audio_norm.unsqueeze(0)
#         spec_filename = filename.replace(".wav", ".spec.pt")
#         if self.use_mel_spec_posterior:
#             spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
#         if os.path.exists(spec_filename):
#             spec = torch.load(spec_filename)
#         else:
#             if self.use_mel_spec_posterior:
#                 """TODO : (need verification)
#                 if linear spec exists convert to
#                 mel from existing linear spec (uncomment below lines)"""
#                 # if os.path.exists(filename.replace(".wav", ".spec.pt")):
#                 #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
#                 #     spec = spec_to_mel_torch(
#                 #         torch.load(filename.replace(".wav", ".spec.pt")),
#                 #         self.filter_length, self.n_mel_channels, self.sampling_rate,
#                 #         self.hparams.mel_fmin, self.hparams.mel_fmax)
#                 spec = mel_spectrogram_torch(
#                     audio_norm,
#                     self.filter_length,
#                     self.n_mel_channels,
#                     self.sampling_rate,
#                     self.hop_length,
#                     self.win_length,
#                     self.hparams.mel_fmin,
#                     self.hparams.mel_fmax,
#                     center=False,
#                 )
#             else:
#                 spec = spectrogram_torch(
#                     audio_norm,
#                     self.filter_length,
#                     self.sampling_rate,
#                     self.hop_length,
#                     self.win_length,
#                     center=False,
#                 )
#             spec = torch.squeeze(spec, 0)
#             torch.save(spec, spec_filename)
#         return spec, audio_norm

#     def get_text(self, text):
#         if self.cleaned_text:
#             text_norm = cleaned_text_to_sequence(text)
#         else:
#             text_norm = text_to_sequence(text, self.text_cleaners)
#         if self.add_blank:
#             text_norm = intersperse(text_norm, 0)
#         text_norm = torch.LongTensor(text_norm)
#         return text_norm

#     def __getitem__(self, index):
#         return self.get_audio_text_pair(self.audiopaths_and_text[index])

#     def __len__(self):
#         return len(self.audiopaths_and_text)


# class TextAudioCollate:
#     """Zero-pads model inputs and targets"""

#     def __init__(self, return_ids=False):
#         self.return_ids = return_ids

#     def __call__(self, batch):
#         """Collate's training batch from normalized text and aduio
#         PARAMS
#         ------
#         batch: [text_normalized, spec_normalized, wav_normalized]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         _, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
#         )

#         max_text_len = max([len(x[0]) for x in batch])
#         max_spec_len = max([x[1].size(1) for x in batch])
#         max_wav_len = max([x[2].size(1) for x in batch])

#         text_lengths = torch.LongTensor(len(batch))
#         spec_lengths = torch.LongTensor(len(batch))
#         wav_lengths = torch.LongTensor(len(batch))

#         text_padded = torch.LongTensor(len(batch), max_text_len)
#         spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
#         wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
#         text_padded.zero_()
#         spec_padded.zero_()
#         wav_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             row = batch[ids_sorted_decreasing[i]]

#             text = row[0]
#             text_padded[i, : text.size(0)] = text
#             text_lengths[i] = text.size(0)

#             spec = row[1]
#             spec_padded[i, :, : spec.size(1)] = spec
#             spec_lengths[i] = spec.size(1)

#             wav = row[2]
#             wav_padded[i, :, : wav.size(1)] = wav
#             wav_lengths[i] = wav.size(1)

#         if self.return_ids:
#             return (
#                 text_padded,
#                 text_lengths,
#                 spec_padded,
#                 spec_lengths,
#                 wav_padded,
#                 wav_lengths,
#                 ids_sorted_decreasing,
#             )
#         return (
#             text_padded,
#             text_lengths,
#             spec_padded,
#             spec_lengths,
#             wav_padded,
#             wav_lengths,
#         )