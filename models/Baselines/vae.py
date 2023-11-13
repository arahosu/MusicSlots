import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

import os
import math
from itertools import chain

from models.shared.classifier import Classifier
from models.shared.encoder import SimpleEncoder
from models.shared.decoder import SimpleDecoder
from models.shared.metrics import compute_accuracy, multilabel_acc

from models.Baselines.utils import get_distributed_labels

from datasets.bach_chorales_utils import plot_multiple_spectrograms, \
    plot_spectrogram, compare_spectrograms


class VanillaVAE(LightningModule):
    """
    Baseline Vanilla Gaussian VAE 
    """
    def __init__(self,
                 in_channels: int,
                 resolution: list,
                 lr: float,
                 grad_clip_val: float,
                 latent_dim: int,
                 kld_weight:float,
                 classifier_lr: float,
                 stride: tuple):

        super(VanillaVAE, self).__init__()

        self.save_hyperparameters()

        self.encoder = SimpleEncoder(
            self.hparams.in_channels, self.hparams.latent_dim,
            kernel_size=(5, 5), num_encoder_layers=4, stride=stride)

        if isinstance(stride, int):
            stride = (stride, stride)

        freq_downsample = math.pow(stride[0], 4)
        time_downsample = math.pow(stride[1], 4)

        self.encoder_output_size = [int(self.hparams.resolution[0] // freq_downsample),
                                    int(self.hparams.resolution[1] // time_downsample)]
                
        self.decoder = SimpleDecoder(
            self.hparams.in_channels, self.hparams.latent_dim, return_alpha_mask=False,
            decoder_initial_size=self.encoder_output_size)

        self.fc_mu = nn.Linear(
            self.hparams.latent_dim * self.encoder_output_size[0] * self.encoder_output_size[1],
            self.hparams.latent_dim)
        self.fc_var = nn.Linear(
            self.hparams.latent_dim * self.encoder_output_size[0] * self.encoder_output_size[1],
            self.hparams.latent_dim)

        self.decoder_input = nn.Linear(self.hparams.latent_dim,
            self.hparams.latent_dim * 8 * 2)

        self.automatic_optimization = False

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.hparams.kld_weight != 0:
            z = self.reparameterize(mu, log_var)
            return self.decode(z), mu, log_var
        else:
            return self.decode(mu), mu, log_var

    def encode(self, x):
        enc_out = torch.flatten(self.encoder(x), start_dim=1)
        mu = self.fc_mu(enc_out)
        log_var = self.fc_var(enc_out)

        return mu, log_var

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.hparams.latent_dim,
                   8,
                   2)
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def sample(self, num_samples: int):
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        # load optimizers and scheduler
        vae_opt = self.optimizers()
        scheduler = self.lr_schedulers()

        # chord_spec, _, chord_label = batch
        chord_spec = batch[0]

        recon, mu, log_var = self(chord_spec)
        recon_loss = F.mse_loss(recon, chord_spec)
        
        if self.hparams.kld_weight != 0.:
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recon_loss + self.hparams.kld_weight * kld_loss

            self.log_dict(
                {"train_kl_div": -kld_loss}
            )
        else:
            loss = recon_loss

        # optimize autoencoder params
        vae_opt.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        if self.hparams.grad_clip_val != 0:
            self.clip_gradients(
                vae_opt, gradient_clip_val=self.hparams.grad_clip_val, gradient_clip_algorithm="norm")

        vae_opt.step()
        scheduler.step()

        self.log_dict({
                "train_loss": loss,
                "train_recon_loss": recon_loss})

    def evaluate(self, batch, stage=None):
        
        # chord_spec, _, chord_label = batch
        chord_spec = batch[0]

        recon, mu, log_var = self(chord_spec)
        recon_loss = F.mse_loss(recon, chord_spec)

        if self.hparams.kld_weight != 0.:
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recon_loss + self.hparams.kld_weight * kld_loss

            self.log_dict(
                {f"{stage}_kl_div": -kld_loss}
            )
        else:
            loss = recon_loss

        self.log_dict({
            f"{stage}_loss": loss,
            f"{stage}_recon_loss": recon_loss})

        return recon, mu, log_var
    
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.fc_mu.parameters(), 
                  self.fc_var.parameters(), self.decoder_input.parameters(), 
                  self.decoder.parameters()),
            lr=self.hparams.lr
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps)  

        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        optimizer_list = [optimizer]
        scheduler_list = [lr_scheduler]

        return optimizer_list, scheduler_list


class VAE_Classifier(LightningModule):
    def __init__(self,
                 backbone,
                 lr,
                 num_notes,
                 num_instruments,
                 linear: bool = False):

        super(VAE_Classifier, self).__init__()
        
        self.save_hyperparameters(ignore='backbone')
        self.backbone = backbone

        self.latent_dim = self.backbone.hparams.latent_dim
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = Classifier(self.latent_dim,
                                     num_notes,
                                     num_instruments,
                                     linear=linear)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        chord_spec, note_specs, chord_labels, inst_labels = batch

        with torch.no_grad():
            mu, _ = self.backbone.encode(chord_spec)
        
        if self.hparams.num_instruments > 1:
            labels = get_distributed_labels(chord_labels, inst_labels,
                                            self.hparams.num_notes, self.hparams.num_instruments)
        else:
            multi_hot = [(F.one_hot(chord, num_classes=self.hparams.num_notes)).sum(0).unsqueeze(0) 
                      for chord in chord_labels]
            labels = torch.cat(multi_hot)

        logits = self.classifier(mu.detach())

        if self.hparams.num_instruments == 1:
            logits = logits.squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # acc = multilabel_acc(labels.reshape(labels.shape[0], -1), torch.sigmoid(logits.reshape(logits.shape[0], -1)))
        acc = compute_accuracy((torch.sigmoid(logits.reshape(logits.shape[0], -1)) >= 0.5).long(),
                                labels.reshape(labels.shape[0], -1))

        self.log("train_chord_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_chord_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def evaluate(self, batch, stage=None):
        chord_spec, note_specs, chord_labels, inst_labels = batch

        with torch.no_grad():
            mu, _ = self.backbone.encode(chord_spec)

        if self.hparams.num_instruments > 1:
            labels = get_distributed_labels(chord_labels, inst_labels,
                                            self.hparams.num_notes, self.hparams.num_instruments)
        else:
            multi_hot = [(F.one_hot(chord, num_classes=self.hparams.num_notes)).sum(0).unsqueeze(0) 
                      for chord in chord_labels]
            labels = torch.cat(multi_hot)

        logits = self.classifier(mu.detach())

        if self.hparams.num_instruments == 1:
            logits = logits.squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # acc = multilabel_acc(labels.reshape(labels.shape[0], -1), torch.sigmoid(logits.reshape(logits.shape[0], -1)))
        acc = compute_accuracy((torch.sigmoid(logits.reshape(logits.shape[0], -1)) >= 0.5).long(),
                                labels.reshape(labels.shape[0], -1))

        self.log(f"{stage}_chord_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_chord_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        params = self.classifier.parameters()
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.lr
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps)  

        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class VAELoggerCallback(Callback):
    def __init__(self,
                 wandb_logger,
                 num_to_log=1,
                 num_samples=10,
                 **kwargs):
        super(VAELoggerCallback, self).__init__(**kwargs)
        
        self.wandb_logger = wandb_logger
        self.num_to_log = num_to_log
        self.num_samples = num_samples

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""
        
        if batch_idx == 0:
            # chord_specs, _, chord = batch
            chord_specs = batch[0]
            recons, _, _ = pl_module(chord_specs)

            chord_specs = chord_specs[:self.num_to_log]
            recons = recons[:self.num_to_log]

            for i, (chord_spec, recon) in enumerate(zip(chord_specs, recons)):
                trainer.logger.experiment.log({
                    "Sample {} Output Reconstruction".format(i+1): plot_multiple_spectrograms(
                            [chord_spec[0].cpu().numpy(), recon[0].cpu().numpy()],
                            figsize=(8, 8), title=['Input', 'Output Reconstruction'], aspect="equal")})
            
            samples = pl_module.sample(self.num_samples)
            
            if pl_module.hparams.kld_weight != 0.:
                trainer.logger.experiment.log({
                    "Generated Samples": plot_multiple_spectrograms(
                        [sample[0].cpu().numpy() for sample in samples], figsize=(int(32 * self.num_samples / 7), 9), aspect="equal")
                })
    
    def on_test_end(self, trainer, pl_module):
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'wandb')):
            # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))