import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics.functional.classification import accuracy, precision

import math

from models.shared.encoder import SimpleEncoder, ResNet
from models.shared.classifier import Classifier
from models.shared.metrics import compute_accuracy

from models.Baselines.utils import get_distributed_labels

class SupervisedClassifier(LightningModule):
    def __init__(self,
                 in_channels,
                 resolution,
                 lr,
                 backbone,
                 num_notes,
                 num_instruments,
                 stride=(2, 2),
                 hidden_dim=128):
        super(SupervisedClassifier, self).__init__()
        
        self.save_hyperparameters()
        
        if backbone == 'resnet':
            backbone = ResNet(
                self.hparams.in_channels, [2, 2, 2, 2],
                stride=stride,
                num_classes=None)
            
            self.classifier = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                Classifier(512, self.hparams.num_notes, self.hparams.num_instruments)
            )
        
        elif backbone == 'simple':
            freq_downsample = math.pow(stride[0], 4)
            time_downsample = math.pow(stride[1], 4)
            
            self.encoder_output_size = [int(self.hparams.resolution[0] // freq_downsample),
                                        int(self.hparams.resolution[1] // time_downsample)]

            backbone = SimpleEncoder(
                self.hparams.in_channels,
                self.hparams.hidden_dim,
                kernel_size=(5, 5), stride=stride)
            
            self.classifier = nn.Sequential(
                backbone,
                nn.Flatten(1),
                nn.Linear(self.encoder_output_size[0] * self.encoder_output_size[1] * self.hparams.hidden_dim, self.hparams.hidden_dim),
                nn.ReLU(),
                Classifier(self.hparams.hidden_dim, self.hparams.num_notes, self.hparams.num_instruments)
            )
    
    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        chord_spec, note_specs, chord_labels, inst_labels = batch

        if self.hparams.num_instruments > 1:
            labels = get_distributed_labels(chord_labels, inst_labels,
                                            self.hparams.num_notes, self.hparams.num_instruments)
        else:
            multi_hot = [(F.one_hot(chord, num_classes=self.hparams.num_notes)).sum(0).unsqueeze(0) 
                      for chord in chord_labels]
            labels = torch.cat(multi_hot)

        logits = self(chord_spec)
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

        if self.hparams.num_instruments > 1:
            labels = get_distributed_labels(chord_labels, inst_labels,
                                            self.hparams.num_notes, self.hparams.num_instruments)
        else:
            multi_hot = [(F.one_hot(chord, num_classes=self.hparams.num_notes)).sum(0).unsqueeze(0) 
                      for chord in chord_labels]
            labels = torch.cat(multi_hot)

        logits = self(chord_spec)
        if self.hparams.num_instruments == 1:
            logits = logits.squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # acc = multilabel_acc(labels.reshape(labels.shape[0], -1), torch.sigmoid(logits.reshape(logits.shape[0], -1)))
        acc = compute_accuracy((torch.sigmoid(logits.reshape(logits.shape[0], -1)) >= 0.5).long(),
                                labels.reshape(labels.shape[0], -1))

        self.log(f"{stage}_chord_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_chord_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
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

        