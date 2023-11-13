import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import torchaudio.functional as AF

from pytorch_lightning import LightningModule
import pytorch_warmup as warmup

from models.MusicSlots.metrics import best_overlap_mse, best_overlap_iou
from models.shared.decoder import SimpleDecoder
from models.shared.encoder import SimpleEncoder, ResNet18, ResNet34

from datasets.bach_chorales_utils import plot_multiple_spectrograms, \
    plot_spectrogram, compare_spectrograms

import os
from functools import partial
import math
from itertools import chain
import numpy as np

from models.MusicSlots.classifier import Classifier
from torchmetrics.functional.classification import accuracy

class SlotAttention(nn.Module):
    """
    Reference: https://github.com/google-research/google-research/blob/master/slot_attention/model.py
    """
    def __init__(self,
                 num_slots: int,
                 d_slot: int,
                 num_iter: int,
                 d_mlp: int,
                 eps: float,
                 use_implicit: bool,
                 share_slot_init: bool):
        super(SlotAttention, self).__init__()
        
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.d_slot = d_slot
        self.d_mlp = d_mlp
        self.eps = eps
        self.use_implicit = use_implicit
        self.share_slot_init = share_slot_init

        self.norm_inputs  = nn.LayerNorm(d_slot)
        self.norm_slots  = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)
        
        # Params for Gaussian init
        # if shared across all slots
        if self.share_slot_init:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, d_slot))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, d_slot))
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, num_slots, d_slot))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, d_slot))

        # Xavier-glorot init
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_logsigma)

        # Linear maps for the attention module
        self.proj_q = nn.Linear(d_slot, d_slot, bias = False)
        self.proj_k = nn.Linear(d_slot, d_slot, bias = False)
        self.proj_v = nn.Linear(d_slot, d_slot, bias = False)
        
        # Slot update functions using MLP and GRU
        self.rnn = nn.GRUCell(d_slot, d_slot)

        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_mlp),
            nn.ReLU(inplace = True),
            nn.Linear(d_mlp, d_slot))
    
    def step(self, slots, k, v, return_attn: bool = False):
        if isinstance(slots, tuple):
            slots, _ = slots
        # get the shape of input slots
        b, n, d = slots.shape

        # Implicit gradient approximation using 1st-order Neumann approximation
        # See: https://arxiv.org/abs/2207.00787 (Chang et. al, 2022)
        if self.use_implicit:
            slots = slots.detach()

        # compute assignments given slots
        q = self.proj_q(self.norm_slots(slots))

        attn = F.softmax(torch.einsum('bkd, bqd -> bkq', q, k), dim = -1) * self.d_slot ** -0.5
        attn = attn / torch.sum(attn + self.eps, dim = -2, keepdim = True)
        
        # update slots given assignments
        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.rnn(
            updates.reshape(-1, self.d_slot),
            slots.reshape(-1, self.d_slot))
        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_mlp(slots))
        
        if return_attn:
            return slots, attn

        return slots

    def iterate(self, f, x):
        for _ in range(self.num_iter):
            x = f(x)
        return x

    def forward(self, x, num_slots: int = None, return_attn: bool = False):
        b, n, d = x.shape
        num_slots = self.num_slots if num_slots is None else num_slots

        x = self.norm_inputs(x)
        k, v = self.proj_k(x), self.proj_v(x)

        if self.share_slot_init:
            mu = self.slots_mu.expand(b, num_slots, -1)
            sigma = self.slots_logsigma.exp().expand(b, num_slots, -1)
        else:
            mu = self.slots_mu.expand(b, -1, -1)
            sigma = self.slots_logsigma.exp().expand(b, -1, -1)

        # initialize slots
        slots = mu + sigma * torch.randn(mu.shape, device=x.device)
        
        # Multiplie rounds of iterations
        if return_attn:
            slots, attn = self.iterate(lambda z: self.step(z, k, v, return_attn=True), slots)
        else:
            slots = self.iterate(lambda z: self.step(z, k, v), slots)

        if return_attn:
            return slots, attn

        return slots


def return_grid(resolution: list):
    # Build a grid for soft positonal encoding 
    ranges = [torch.linspace(0., 1., steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, axis=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = torch.unsqueeze(grid, axis=0)
    return torch.cat([grid, 1.0 - grid], axis=-1)


class SoftPositionEncoding(nn.Module):

    def __init__(self, hidden_size: int, resolution: list):
        """
        Builds the soft position encoding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: List of integers specifying width and height of grid.
        """
        super(SoftPositionEncoding, self).__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = return_grid(resolution)

    def forward(self, inputs):
        device = inputs.device
        grid = self.embedding(self.grid.to(device))

        return inputs + grid


class SlotAttentionAE(LightningModule):
    """
    Slot Attention AutoEncoder for learning musical objects
    """
    def __init__(self,
                 in_channels: int,
                 resolution: list,
                 lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 num_slots: int,
                 num_iter: int,
                 d_slot: int,
                 d_mlp: int,
                 eps: float,
                 use_implicit: bool,
                 share_slot_init: bool,
                 alpha_mask_type: str,
                 kernel_height: int = 5,
                 num_encoder_layers: int = 4,
                 num_strided_decoder_layers: int = 4,
                 use_deconv: bool = True,
                 grad_clip_val: float = 1.0,
                 stride: tuple = (1, 1),
                 db_thres: float = -30.0,
                 to_db: bool = True,
                 top_db: float = 80.0,
                 encoder_type: str = 'simple',
                 num_to_log: int = 5,
                 logging_epoch_freq: int = 25):
        super(SlotAttentionAE, self).__init__()

        self.save_hyperparameters()

        if self.hparams.encoder_type == 'simple':
            self.encoder_cnn = SimpleEncoder(c_in=self.hparams.in_channels,
                                             c_out=self.hparams.d_slot,
                                             kernel_size=(self.hparams.kernel_height, 5),
                                             num_encoder_layers=self.hparams.num_encoder_layers,
                                             stride = stride)
            c_out = self.hparams.d_slot

            freq_downsample = math.pow(stride[0], 4)
            time_downsample = math.pow(stride[1], 4)

            self.encoder_pos = SoftPositionEncoding(
                hidden_size=c_out,
                resolution=[int(self.hparams.resolution[0] // freq_downsample),
                            int(self.hparams.resolution[1] // time_downsample)])

        else:
            if self.hparams.encoder_type == 'resnet18':
                self.encoder_cnn = ResNet18(
                    c_in=self.hparams.in_channels, stride=stride)
            elif self.hparams.encoder_type == 'resnet34':
                self.encoder_cnn = ResNet34(
                    c_in=self.hparams.in_channels, stride=stride)
            
            c_out = 512
            freq_downsample = math.pow(stride[0], 3)
            time_downsample = math.pow(stride[1], 3)

            self.encoder_pos = SoftPositionEncoding(
                hidden_size=c_out,
                resolution=[int(self.hparams.resolution[0] // freq_downsample),
                            int(self.hparams.resolution[1] // time_downsample)])
        
        self.decoder_initial_size = [int(self.hparams.resolution[0] // math.pow(2, self.hparams.num_strided_decoder_layers)),
                                     int(self.hparams.resolution[1] // math.pow(2, self.hparams.num_strided_decoder_layers))]
        self.decoder_cnn = SimpleDecoder(c_in=self.hparams.in_channels,
                                         c_out=self.hparams.d_slot,
                                         decoder_initial_size=self.decoder_initial_size,
                                         num_strided_decoder_layers=self.hparams.num_strided_decoder_layers,
                                         return_alpha_mask=False if self.hparams.alpha_mask_type == 'linear' else True,
                                         use_deconv=self.hparams.use_deconv)
        self.decoder_pos = SoftPositionEncoding(hidden_size=self.hparams.d_slot,
                                                resolution=self.decoder_initial_size)

        # conv1x1 layers implemented as MLP 
        self.fc1 = nn.Linear(c_out, c_out)
        self.fc2 = nn.Linear(c_out, self.hparams.d_slot)

        self.slot_attention = SlotAttention(
            num_slots=self.hparams.num_slots,
            d_slot=self.hparams.d_slot,
            num_iter = self.hparams.num_iter,
            eps = self.hparams.eps, 
            d_mlp = self.hparams.d_mlp,
            use_implicit = self.hparams.use_implicit,
            share_slot_init = self.hparams.share_slot_init)

        self.automatic_optimization = False
    
    def forward(self, input):
        # CNN encoder with soft positional encoding
        device = input.device
        x = self.encoder_cnn(input)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Slot Attention Module
        slots = self.slot_attention(x)

        # Broadcast slot features to a 2D grid and collapse slot dimension
        slots_reshaped = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_reshaped = slots_reshaped.repeat(
            (1, self.decoder_initial_size[0],
             self.decoder_initial_size[1], 1))
        
        # CNN decoder with soft positional encoding
        x = self.decoder_pos(slots_reshaped)
        x = x.permute(0,3,1,2)
        x = self.decoder_cnn(x)
        x = x[..., :self.hparams.resolution[0], :self.hparams.resolution[1]]
        x = x.permute(0,2,3,1)

        # If we use alpha masks, undo combination of slot and batch dimension; split alpha masks.
        if self.hparams.alpha_mask_type != 'linear':
            recons, masks = x.reshape(
                input.shape[0], -1,
                x.shape[1], x.shape[2],
                x.shape[3]).split([self.hparams.in_channels,1], dim=-1)
        # If not, return only the reconstructions and alpha masks to be empty
        else:
            recons = x.reshape(input.shape[0], -1, x.shape[1], x.shape[2], x.shape[3])
            masks = None

        if self.hparams.to_db:
            # convert db scale to power scale
            recons = AF.DB_to_amplitude(recons, 1.0, 1.0)

        # Normalize alpha masks over slots.
        if self.hparams.alpha_mask_type == 'softmax':
            masks = nn.Softmax(dim=1)(masks)
            recon_combined = torch.sum(recons * masks, dim=1)
        elif self.hparams.alpha_mask_type == 'sigmoid':
            masks = nn.Sigmoid()(masks)
            recon_combined = torch.sum(recons * masks, dim=1)
        elif self.hparams.alpha_mask_type == 'linear':
            recon_combined = torch.sum(recons, dim=1)
        recon_combined = recon_combined.permute(0,3,1,2)

        if self.hparams.to_db:
            # convert back from power scale to db_scale
            recon_combined = AF.amplitude_to_DB(
                recon_combined, 10, 1e-10, 0, top_db=self.hparams.top_db)

        return recon_combined, recons, masks, slots

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # load optimizers and scheduler
        slot_attention_opt = self.optimizers()
        scheduler = self.lr_schedulers()

        # chord_specs, note_specs, chord_label, inst_label = batch
        chord_specs = batch[0]
        note_specs = batch[1]

        recon_combined, recons, masks, slots = self(chord_specs)

        recon_loss = F.mse_loss(recon_combined, chord_specs)
        
        if self.hparams.alpha_mask_type != 'linear':
            slot_recons = recons.squeeze() * masks.squeeze()
            slot_recons = AF.amplitude_to_DB(
                slot_recons, 10, 1e-10, 0, top_db=self.hparams.top_db)
        else:
            slot_recons = AF.amplitude_to_DB(
                recons.squeeze(), 10, 1e-10, 0, top_db=self.hparams.top_db)

        slot_recon_masks = (slot_recons >= self.hparams.db_thres).long()
        note_masks = [note_spec >= self.hparams.db_thres for note_spec in note_specs]

        best_iou = best_overlap_iou(note_masks, slot_recon_masks)
        best_mse, best_match_idx = best_overlap_mse(note_specs, slot_recons, return_idx=True)
    
        self.log_dict({"train_loss": recon_loss,
                        "train_best_ov_mse": best_mse,
                        "train_best_ov_iou": best_iou})
        
        # optimize slot attention autoencoder 
        slot_attention_opt.zero_grad()
        self.manual_backward(recon_loss)

        # clip gradients
        self.clip_gradients(
            slot_attention_opt,
            gradient_clip_val=self.hparams.grad_clip_val,
            gradient_clip_algorithm="norm")
        
        slot_attention_opt.step()
        
        if self.hparams.warmup_steps != 0:
            with scheduler.warmup_scheduler.dampening():
                scheduler.step()
        else:
            scheduler.step()

    def evaluate(self, batch, batch_idx, stage=None):
        # chord_specs, note_specs, chord_label, inst_label = batch
        chord_specs = batch[0]
        note_specs = batch[1]

        recon_combined, recons, masks, slots = self(chord_specs)
        recon_loss = F.mse_loss(recon_combined, chord_specs)
        
        if self.hparams.alpha_mask_type != 'linear':
            slot_recons = recons.squeeze() * masks.squeeze()
            slot_recons = AF.amplitude_to_DB(
                slot_recons, 10, 1e-10, 0, top_db=self.hparams.top_db)
        else:
            slot_recons = AF.amplitude_to_DB(
                recons.squeeze(), 10, 1e-10, 0, top_db=self.hparams.top_db)

        slot_recon_masks = (slot_recons >= self.hparams.db_thres).long()
        note_masks = [note_spec >= self.hparams.db_thres for note_spec in note_specs]

        best_iou = best_overlap_iou(note_masks, slot_recon_masks)
        best_mse, best_match_idx = best_overlap_mse(note_specs, slot_recons, return_idx=True)
    
        self.log_dict({f"{stage}_loss": recon_loss,
                       f"{stage}_best_ov_mse": best_mse,
                       f"{stage}_best_ov_iou": best_iou})

        # Log predictions to Wandb.ai if trainer logger is WandbLogger
        if stage == "val" and self.trainer.logger.__class__.__name__ == 'WandbLogger':
            if batch_idx == 0 and self.current_epoch % self.hparams.logging_epoch_freq == 0:
                for i in range(self.hparams.num_to_log):
                    slot_recon = slot_recons[i]
                    matching_idx = best_match_idx[i]
                    non_matching_idx = [i for i in range(slots.shape[1]) if i not in matching_idx]

                    specs1 = [chord_specs[i, 0].cpu().numpy().squeeze()] + [note.cpu().numpy() for note in note_specs[i]]
                    specs2 = [recon_combined[i, 0].cpu().numpy().squeeze()] + \
                        [sl.cpu().numpy() for sl in slot_recon[np.array(matching_idx)]] + \
                        [sl.cpu().numpy() for sl in slot_recon[np.array(non_matching_idx)]]

                    note_masks = [note.cpu().numpy() >= self.hparams.db_thres for note in note_specs[i]]
                    matching_masks = [sl.cpu().numpy() >= self.hparams.db_thres for sl in slot_recon[np.array(matching_idx)]]
                    non_matching_masks = [sl.cpu().numpy() >= self.hparams.db_thres for sl in slot_recon[np.array(non_matching_idx)]]

                    all_masks = matching_masks + non_matching_masks

                    self.trainer.logger.experiment.log(
                        {"Sample {} Reconstructions".format(i+1): compare_spectrograms(
                            specs1, specs2, figsize=(int(12 * self.hparams.num_slots / 7), 9), ylabel='Mel Bins', colorbar_label='Decibels / dB',
                            subtitle1=['Chord Spec.'] + ['Note {}'.format(i + 1) for i in range(len(note_specs[i]))],
                            subtitle2=['Full Recon.'] + ['Slot {}'.format(j + 1) for j in range(len(slot_recon))]),
                         "Sample {} Masks".format(i+1): compare_spectrograms(
                            note_masks, all_masks, figsize=(int(14 * self.hparams.num_slots / 7), 9),
                            ylabel='Mel Bins', colorbar_label=None,
                            subtitle1=['Note {} Mask'.format(i + 1) for i in range(len(note_specs[i]))],
                            subtitle2=['Slot {} Mask '.format(j + 1) for j in range(self.hparams.num_slots)])
                    })
                    
                    if self.hparams.alpha_mask_type != 'linear':
                        self.trainer.logger.experiment.log(
                            {"Sample {} Alpha Masks".format(i+1): plot_multiple_spectrograms(
                                [alpha_mask.cpu().numpy() for alpha_mask in masks[i]],
                                 figsize=(int(14 * self.hparams.num_slots / 7), 9), 
                                 title=['Slot {}'.format(j + 1) for j in range(self.hparams.num_slots)],
                                 aspect="equal", colorbar_label=None)
                            }
                        )
        
        return recon_combined, recons, masks, slots

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # self.parameters(),
            chain(self.encoder_cnn.parameters(),
                  self.encoder_pos.parameters(),
                  self.fc1.parameters(),
                  self.fc2.parameters(),
                  self.slot_attention.parameters(),
                  self.decoder_pos.parameters(),
                  self.decoder_cnn.parameters()),
            lr=self.hparams.lr
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.decay_steps)
        if self.hparams.warmup_steps != 0:
            lr_scheduler.warmup_scheduler = warmup.LinearWarmup(
                optimizer, warmup_period=self.hparams.warmup_steps)

        optimizer_list = [optimizer]
        scheduler_list = [lr_scheduler]

        return optimizer_list, scheduler_list

    def on_test_end(self):
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'wandb')):
            # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))