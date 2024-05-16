from typing import Any, Dict, Tuple
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.utils.instantiators import instantiate_module
import src.models.modules.commons as commons
from src.data.components.mel_processing import mel_spectrogram_torch
from src.models.modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from src.utils.logging_utils import summarize, plot_spectrogram_to_numpy, plot_alignment_to_numpy


class VITS2LitModule(LightningModule):
    """

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net_g_cfg: DictConfig,
        net_d_cfg: DictConfig,
        net_dur_disc_cfg: DictConfig,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        optimizer_dur_disc: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler,
        scheduler_d: torch.optim.lr_scheduler,
        scheduler_dur_disc: torch.optim.lr_scheduler,
        mel_cfg: DictConfig,
        c_mel: int,
        c_kl: float,
    ) -> None:
        """Initialize a `SFTTSLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.net_g = instantiate_module(net_g_cfg)
        self.net_d = instantiate_module(net_d_cfg)
        self.use_dur_disc = net_dur_disc_cfg.pop("use")
        if self.use_dur_disc:
            self.net_dur_disc = instantiate_module(net_dur_disc_cfg)
        else:
            self.hparams.optimizer_dur_disc = None
            self.hparams.scheduler_dur_disc = None
        
    

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the model `self.net_g`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = batch['text']
        x_lengths = batch['text_len']
        spec = batch['spec']
        spec_lengths = batch['spec_len']
        speakers = batch['sid']
        return self.net_g.forward(x, x_lengths, spec, spec_lengths, speakers)
        
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        
        if self.use_dur_disc:
            optim_g, optim_d, optim_dur_disc = self.optimizers()
            sch_g, sch_d, sch_dur_disc = self.lr_schedulers()
        else:
            optim_g, optim_d = self.optimizers()
            sch_g, sch_d = self.lr_schedulers()
            
        self.net_g.update_current_mas_noise_scale(self.global_step)
        self.log('global_step', self.global_step, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        
        # y, y_hat, y_mel, y_hat_mel, mel
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_x, logw, logw_),
        ) = self.forward(batch)
        
        mel = batch['spec']
        y_mel = commons.slice_segments(mel, ids_slice, self.hparams.net_g_cfg.segment_size)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.mel_cfg.filter_length,
            self.hparams.mel_cfg.n_mel_channels,
            self.hparams.mel_cfg.sampling_rate,
            self.hparams.mel_cfg.hop_length,
            self.hparams.mel_cfg.win_length,
            self.hparams.mel_cfg.mel_fmin,
            self.hparams.mel_cfg.mel_fmax,
        )
        y = commons.slice_segments(
            batch['wav'], ids_slice * self.hparams.mel_cfg.hop_length, 
            self.hparams.net_g_cfg.segment_size * self.hparams.mel_cfg.hop_length
        )  # slice
        
        
        # Train Discriminator
        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc
        
        # self.train_d_loss(loss_disc_all)
        # self.log("train/d_loss", self.train_d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/d_loss", loss_disc_all, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        [self.log(f"train/d_r_{i}", v, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True) for i, v in enumerate(losses_disc_r)]
        [self.log(f"train/d_g_{i}", v, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True) for i, v in enumerate(losses_disc_g)]
        self.manual_backward(loss_disc_all)
        self.clip_gradients(optim_d, gradient_clip_val=None, gradient_clip_algorithm='norm')
        optim_d.step()
        optim_d.zero_grad()
        self.untoggle_optimizer(optim_d)
        
        #  Train Duration Discriminator
        if self.use_dur_disc:
            self.toggle_optimizer(optim_dur_disc)   
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(
                hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
            )
            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                y_dur_hat_r, 
                y_dur_hat_g
            )
            loss_dur_disc_all = loss_dur_disc
            
            # self.train_dur_d_loss(loss_dur_disc_all)
            # self.log("train/dur_d_loss", self.train_dur_d_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/dur_d_loss", loss_dur_disc_all, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.manual_backward(loss_dur_disc_all)
            self.clip_gradients(optim_dur_disc, gradient_clip_val=None, gradient_clip_algorithm='norm')
            optim_dur_disc.step()
            optim_dur_disc.zero_grad()
            self.untoggle_optimizer(optim_dur_disc)
            
        # Train Generator
        self.toggle_optimizer(optim_g)
        
        _, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        if self.use_dur_disc:
            _, y_dur_hat_g = self.net_dur_disc(hidden_x, x_mask, logw_, logw)
        
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_g, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_g + loss_fm + loss_mel + loss_dur + loss_kl
        if self.use_dur_disc:
            loss_dur_g, losses_dur_gen = generator_loss(y_dur_hat_g)
            loss_gen_all += loss_dur_g
            self.log("train/dur_g_loss", loss_dur_g, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            
        [self.log(f"train/g_loss_{i}", v, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True) for i, v in enumerate(losses_gen)]
        self.log("train/fm_loss", loss_fm, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/mel_loss", loss_mel, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/dur_loss", loss_dur, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl_loss", loss_kl, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/gen_all_loss", loss_gen_all, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.manual_backward(loss_gen_all)
        self.clip_gradients(optim_g, gradient_clip_val=None, gradient_clip_algorithm='norm')
        optim_g.step()
        optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)
        
            
        # log mel, attn every 1 epoch
        if self.trainer.is_last_batch and self.logger is not None:
            image_dict = {
                    "slice/mel_org": plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
            summarize(
                writer=self.logger.experiment,
                global_step=self.global_step,
                images=image_dict,
            )
            
        # scheduler step every 1 epoch
        if self.trainer.is_last_batch:
            sch_g.step()
            sch_d.step()
            if self.use_dur_disc:
                sch_dur_disc.step()
        

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        if batch_idx == 0 and self.logger is not None:
            x = batch['text'][:1]
            x_lengths = batch['text_len'][:1]
            mel = batch['spec'][:1]
            speakers = batch['sid'][:1]
            y = batch['wav'][:1]
            y_lengths = batch['wav_len'][:1]
            
            y_hat, attn, mask, *_ = self.net_g.infer(
                x, x_lengths, speakers, max_len=1000
            )
            y_hat_lengths = mask.sum([1, 2]).long() * self.hparams.mel_cfg.hop_length

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                self.hparams.mel_cfg.filter_length,
                self.hparams.mel_cfg.n_mel_channels,
                self.hparams.mel_cfg.sampling_rate,
                self.hparams.mel_cfg.hop_length,
                self.hparams.mel_cfg.win_length,
                self.hparams.mel_cfg.mel_fmin,
                self.hparams.mel_cfg.mel_fmax,
            )
            image_dict = {
                "gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
            }
            audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
            image_dict.update(
                {"gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
            )
            audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

            summarize(
                writer=self.logger.experiment,
                global_step=self.global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sampling_rate=self.hparams.mel_cfg.sampling_rate,
            )
                
                
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_g = self.hparams.optimizer_g(params=self.net_g.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.net_d.parameters())
        if self.use_dur_disc:
            optimizer_dur_disc = self.hparams.optimizer_d(params=self.net_dur_disc.parameters())
            
        if self.hparams.scheduler_g is not None:
            scheduler_g = self.hparams.scheduler_g(
                optimizer=optimizer_g,
            )
            scheduler_d = self.hparams.scheduler_d(
                optimizer=optimizer_d,
            )
            if self.use_dur_disc:
                scheduler_dur_disc = self.hparams.scheduler_dur_disc(
                    optimizer=optimizer_dur_disc,
                )
            
        if self.use_dur_disc:
            if self.hparams.scheduler_g is not None:
                return (
                    {
                        "optimizer": optimizer_g,
                        "lr_scheduler": {
                            "scheduler": scheduler_g,
                            "interval": "epoch",
                        },
                    }, 
                    {
                        "optimizer": optimizer_d,
                        "lr_scheduler": {
                            "scheduler": scheduler_d,
                            "interval": "epoch",
                        },
                    }, 
                    {
                        "optimizer": optimizer_dur_disc,
                        "lr_scheduler": {
                            "scheduler": scheduler_dur_disc,
                            "interval": "epoch",
                        },
                    }
                )
            else:
                return (
                    {"optimizer": optimizer_g}, 
                    {"optimizer": optimizer_d}, 
                    {"optimizer": optimizer_dur_disc}
                ) 
        else:
            if self.hparams.scheduler_g is not None:
                return (
                    {
                        "optimizer": optimizer_g,
                        "lr_scheduler": {
                            "scheduler": scheduler_g,
                            "interval": "epoch",
                        },
                    }, 
                    {
                        "optimizer": optimizer_d,
                        "lr_scheduler": {
                            "scheduler": scheduler_d,
                            "interval": "epoch",
                        },
                    }
                )
            else:
                return (
                    {"optimizer": optimizer_g}, 
                    {"optimizer": optimizer_d}, 
                )


if __name__ == "__main__":
    _ = VITS2LitModule(None, None, None, None, None, None)
