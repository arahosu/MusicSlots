import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.utilities import rank_zero_warn

from models.shared.classifier import Classifier, MultiHeadClassifier
from models.shared.metrics import compute_accuracy

from models.MusicSlots.metrics import best_overlap_mse

def chord_level_instrument_note_acc(note_preds,
                                    note_gts,
                                    inst_preds,
                                    inst_gts):

    """
    Calculate the accuracy of predicted notes and instruments at the chord level.

    Parameters:
    - note_preds: List of predicted notes for each chord.
    - note_gts: List of actual (ground truth) notes for each chord.
    - inst_preds: List of predicted instruments for each chord.
    - inst_gts: List of actual (ground truth) instruments for each chord.

    Returns:
    - Overall chord accuracy (both note and instrument are correct).
    """
    
    assert len(note_gts) == len(inst_gts)
    
    # Overall chord accuracy (both note and instrument)
    chord_acc = float(sum([((inst_pred == inst_gt).all() and (note_pred == note_gt).all())
                      for note_pred, inst_pred, note_gt, inst_gt in 
                      zip(note_preds, inst_preds, note_gts, inst_gts)]))
    chord_acc /= len(inst_gts)

    return 100 * chord_acc


class SlotClassifier(LightningModule):
    def __init__(self,
                 backbone,
                 lr,
                 num_notes,
                 num_instruments,
                 linear: bool = False):

        super(SlotClassifier, self).__init__()
        
        self.save_hyperparameters(ignore='backbone')
        
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.d_slot = self.backbone.hparams.d_slot
        self.alpha_mask_type = self.backbone.hparams.alpha_mask_type
        self.top_db = self.backbone.hparams.top_db

        if self.hparams.num_instruments > 1:
            self.classifier = MultiHeadClassifier(self.d_slot,
                                                  self.hparams.num_notes,
                                                  self.hparams.num_instruments,
                                                  linear=self.hparams.linear)
        else:
            self.classifier = Classifier(self.d_slot,
                                         self.hparams.num_notes,
                                         self.hparams.num_instruments,
                                         linear=self.hparams.linear)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        chord_spec, note_specs, chord_labels, inst_labels = batch

        with torch.no_grad():
            recon_combined, recons, masks, slots = self.backbone(
                chord_spec)

        if self.alpha_mask_type != 'linear':
            slot_recons = recons.detach().squeeze() * masks.detach().squeeze()
            slot_recons = AF.amplitude_to_DB(
                slot_recons, 10, 1e-10, 0, top_db=self.top_db)
        else:
            slot_recons = AF.amplitude_to_DB(
                recons.detach().squeeze(), 10, 1e-10, 0, top_db=self.top_db)

        # collect the ground-truth labels for notes
        notes = torch.cat([chord for chord in chord_labels])

        # find the best matching note specs for each slot 
        _, idx = best_overlap_mse(note_specs, slot_recons, return_idx=True)
        best_match_slots = torch.cat([slot[i] for i, slot in zip(idx, slots.detach())])
        
        # get the predictions 
        if self.hparams.num_instruments > 1:
            note_logits, inst_logits = self.classifier(best_match_slots)
            
            # compute cross entropy for instruments
            inst_orders = torch.cat([inst_label for inst_label in inst_labels])
            inst_loss = F.cross_entropy(inst_logits, inst_orders)
            
            # compute instrument accuracy
            inst_preds = [self.classifier(slot[i])[1].argmax(-1) for i, slot in zip(idx, slots)]
            chord_inst_acc = compute_accuracy(inst_preds, inst_labels)

            # get note predictions
            note_preds = [self.classifier(slot[i])[0].argmax(-1) for i, slot in zip(idx, slots)]

            # compute cross entropy for notes
            note_loss = F.cross_entropy(note_logits, notes)

            # compute note accuracy
            chord_note_acc = compute_accuracy(note_preds, chord_labels)

            # compute overall chord prediction accuracy
            chord_acc = chord_level_instrument_note_acc(
                note_preds, chord_labels, inst_preds, inst_labels)

            # overall loss
            total_loss = note_loss + inst_loss

            # log metrics
            self.log("train_inst_loss", inst_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_note_loss", note_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

            self.log("train_inst_acc", chord_inst_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_note_acc", chord_note_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", chord_acc, on_step=False, on_epoch=True, prog_bar=True)

        else:
            # remove last dimension
            note_logits = self.classifier(best_match_slots).squeeze()

            # compute cross entropy for notes
            note_loss = F.cross_entropy(note_logits, notes)

            # get note predictions
            note_preds = [self.classifier(slot[i]).squeeze().argmax(-1) for i, slot in zip(idx, slots)]

            # compute note accuracy
            chord_note_acc = compute_accuracy(note_preds, chord_labels)

            # log metrics
            self.log("train_note_loss", note_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_note_acc", chord_note_acc, on_step=False, on_epoch=True, prog_bar=True)

            total_loss = note_loss

        return total_loss

    def evaluate(self, batch, stage=None):
        chord_spec, note_specs, chord_labels, inst_labels = batch

        with torch.no_grad():
            recon_combined, recons, masks, slots = self.backbone(
                chord_spec)

        if self.alpha_mask_type != 'linear':
            slot_recons = recons.detach().squeeze() * masks.detach().squeeze()
            slot_recons = AF.amplitude_to_DB(
                slot_recons, 10, 1e-10, 0, top_db=self.top_db)
        else:
            slot_recons = AF.amplitude_to_DB(
                recons.detach().squeeze(), 10, 1e-10, 0, top_db=self.top_db)

        # collect the ground-truth labels for notes
        notes = torch.cat([chord for chord in chord_labels])

        # find the best matching note specs for each slot 
        _, idx = best_overlap_mse(note_specs, slot_recons, return_idx=True)
        best_match_slots = torch.cat([slot[i] for i, slot in zip(idx, slots.detach())])
        
        # get the predictions 
        if self.hparams.num_instruments > 1:
            note_logits, inst_logits = self.classifier(best_match_slots)
            
            # compute cross entropy for instruments
            inst_orders = torch.cat([inst_label for inst_label in inst_labels])
            inst_loss = F.cross_entropy(inst_logits, inst_orders)
            
            # compute instrument accuracy
            inst_preds = [self.classifier(slot[i])[1].argmax(-1) for i, slot in zip(idx, slots)]
            chord_inst_acc = compute_accuracy(inst_preds, inst_labels)

            # get note predictions
            note_preds = [self.classifier(slot[i])[0].argmax(-1) for i, slot in zip(idx, slots)]

            # compute cross entropy for notes
            note_loss = F.cross_entropy(note_logits, notes)

            # compute note accuracy
            chord_note_acc = compute_accuracy(note_preds, chord_labels)

            # compute overall chord prediction accuracy
            chord_acc = chord_level_instrument_note_acc(
                note_preds, chord_labels, inst_preds, inst_labels)

            # overall loss
            total_loss = note_loss + inst_loss

            # log metrics
            self.log(f"{stage}_inst_loss", inst_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_note_loss", note_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

            self.log(f"{stage}_inst_acc", chord_inst_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_note_acc", chord_note_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc", chord_acc, on_step=False, on_epoch=True, prog_bar=True)

        else:
            # remove last dimension
            note_logits = self.classifier(best_match_slots).squeeze()

            # compute cross entropy for notes
            note_loss = F.cross_entropy(note_logits, notes)

            # get note predictions
            note_preds = [self.classifier(slot[i]).squeeze().argmax(-1) for i, slot in zip(idx, slots)]

            # compute note accuracy
            chord_note_acc = compute_accuracy(note_preds, chord_labels)

            # log metrics
            self.log(f"{stage}_note_loss", note_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_note_acc", chord_note_acc, on_step=False, on_epoch=True, prog_bar=True)

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

# if __name__ == '__main__':
    
#     x = torch.randn((1, 5, 128))
#     y = torch.randint(low=0, high=54, size=(1, 4)).long()
#     print(y.shape)

#     classifier = Classifier(128, 54, 128)
#     out = classifier(x)
#     print(out.shape)

#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(out.view(-1, classifier.num_midi_tokens), y.flatten())
#     print(loss)

# chords_multihot = torch.cat([(F.one_hot(chord.to(pl_module.device), num_classes=self.num_classes)).sum(0).unsqueeze(0) for chord in chords])
# logits = self.mlp_classifier(slot_recons)
# logits_multihot = torch.sigmoid(logits)
# assert chords_multihot.shape == logits_multihot.shape
# loss = F.binary_cross_entropy(logits, chords_oh)
# acc = accuracy(logits_multihot, chords_multihot, task='multilabel', num_classes=self.num_classes)