from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.tts_dataset import TextAudioSpeakerDataset, TextAudioSpeakerCollate



class TTSDataModule(LightningDataModule):
    """Aug LightningDataModule.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            training_files: str,
            validation_files: str,
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
            win_length: int = 512,
            n_mel_channels: int = 80,
            mel_fmin: float = 0.0,
            mel_fmax: Optional[float] = None,
            add_blank: bool = False,
            n_speakers: int = 3000,
            use_ext_spk_emb: bool = False,
            random_seed: int = 42,
            return_ids: bool = False,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = TextAudioSpeakerDataset(
                audiopaths_sid_text=self.hparams.training_files,
                use_mel_posterior_encoder=self.hparams.use_mel_posterior_encoder,
                cleaned_text=self.hparams.cleaned_text,
                text_cleaners=self.hparams.text_cleaners,
                min_text_len=self.hparams.min_text_len,
                max_text_len=self.hparams.max_text_len,
                max_wav_value=self.hparams.max_wav_value,
                min_audio_len=self.hparams.min_audio_len,
                sampling_rate=self.hparams.sampling_rate,
                filter_length=self.hparams.filter_length,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                n_mel_channels=self.hparams.n_mel_channels,
                mel_fmin=self.hparams.mel_fmin,
                mel_fmax=self.hparams.mel_fmax,
                add_blank=self.hparams.add_blank,
                n_speakers=self.hparams.n_speakers,
                use_ext_spk_emb=self.hparams.use_ext_spk_emb,
                random_seed=self.hparams.random_seed,
            )
            
        if not self.data_val:
            self.data_val = TextAudioSpeakerDataset(
                audiopaths_sid_text=self.hparams.validation_files,
                use_mel_posterior_encoder=self.hparams.use_mel_posterior_encoder,
                cleaned_text=self.hparams.cleaned_text,
                text_cleaners=self.hparams.text_cleaners,
                min_text_len=self.hparams.min_text_len,
                max_text_len=self.hparams.max_text_len,
                max_wav_value=self.hparams.max_wav_value,
                min_audio_len=self.hparams.min_audio_len,
                sampling_rate=self.hparams.sampling_rate,
                filter_length=self.hparams.filter_length,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                n_mel_channels=self.hparams.n_mel_channels,
                mel_fmin=self.hparams.mel_fmin,
                mel_fmax=self.hparams.mel_fmax,
                add_blank=self.hparams.add_blank,
                n_speakers=self.hparams.n_speakers,
                use_ext_spk_emb=self.hparams.use_ext_spk_emb,
                random_seed=self.hparams.random_seed,
                test=True
            )
            
        if not self.data_test:
            self.data_test = TextAudioSpeakerDataset(
                audiopaths_sid_text=self.hparams.validation_files,
                use_mel_posterior_encoder=self.hparams.use_mel_posterior_encoder,
                cleaned_text=self.hparams.cleaned_text,
                text_cleaners=self.hparams.text_cleaners,
                min_text_len=self.hparams.min_text_len,
                max_text_len=self.hparams.max_text_len,
                max_wav_value=self.hparams.max_wav_value,
                min_audio_len=self.hparams.min_audio_len,
                sampling_rate=self.hparams.sampling_rate,
                filter_length=self.hparams.filter_length,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                n_mel_channels=self.hparams.n_mel_channels,
                mel_fmin=self.hparams.mel_fmin,
                mel_fmax=self.hparams.mel_fmax,
                add_blank=self.hparams.add_blank,
                n_speakers=self.hparams.n_speakers,
                use_ext_spk_emb=self.hparams.use_ext_spk_emb,
                random_seed=self.hparams.random_seed,
                test=True
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=TextAudioSpeakerCollate(self.hparams),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=TextAudioSpeakerCollate(self.hparams),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=TextAudioSpeakerCollate(self.hparams),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = TTSDataModule()
