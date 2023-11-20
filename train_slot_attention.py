import torch
from torchvision import transforms

from absl import app
from absl import flags
import os
import random
from functools import partial
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from dataset import MusicalObjectDataModule, spec_crop

from models.MusicSlots.slot_attention import SlotAttentionAE
from models.MusicSlots.classifier import SlotClassifier 

# Random seed
flags.DEFINE_integer(
    'seed', 42,
    'random seed')

# Dataset flags
flags.DEFINE_string(
    'dataset', 'jazznet_multi_inst',
    'bach_chorales, jazznet, jazznet_multi_inst, bach_chorales_multi_inst')
flags.DEFINE_integer(
    'num_chords', 1,
    'number of chords in each spectrogram. 1 (default)')
flags.DEFINE_list(
    'instrument_names', ['Yamaha Grand Piano'],
    'List of instruments in the dataset. Either specify instrument name as a string or as an integer index')
flags.DEFINE_string(
    'spec', 'mel',
    'spectrogram representation to use. mel (default) or cqt')
flags.DEFINE_list(
    'img_size', [128, 32],
    'dimension of input spectrogram')
flags.DEFINE_bool(
    'to_db', True,
    'whether to convert the amplitude of the spectrograms to db scale. True (default)')
flags.DEFINE_float(
    'top_db', 80.0,
    'threshold the output at top_db below the peak. 80.0 (default)')

# Accelerator flags
flags.DEFINE_bool(
    'use_gpu', True,
    'set whether to use GPU')
flags.DEFINE_list(
    'device_id', [0], 
    'set which GPU device to use')
flags.DEFINE_bool(
    'use_mixed_precision', True,
    'set whether to use mixed precision fp16 training or not')

# Training flags
flags.DEFINE_integer(
    'num_workers', 8,
    'set the number of workers for the dataloader')
flags.DEFINE_integer(
    'batch_size', 32,
    'set batch size for training.')
flags.DEFINE_float(
    'lr', 3e-04,
    'set learning rate for training')
flags.DEFINE_float(
    'grad_clip_val', 1.0,
    'set value for gradient clipping')
flags.DEFINE_integer(
    'warmup_steps', 10000,
    'Number of warmup steps for learning rate schedule')
flags.DEFINE_integer(
    'max_steps', 100000,
    'Maximum number of steps for training')
flags.DEFINE_integer(
    'decay_steps', 500000,
    'Number of steps to decay the learning rate')
flags.DEFINE_float(
    'classifier_lr', 1e-03,
    'set learning rate for classifiers. 1e-03 (default)')
flags.DEFINE_integer(
    'classifier_steps', 10000,
    'set number of steps for training the downstream classifier')
flags.DEFINE_bool(
    'use_linear_classifier', True,
    'whether to train a downstream linear classifier (Default) or MLP classifier')

# Model args
flags.DEFINE_integer(
    'num_slots', 7,
    'Number of slots in the slot attention model')
flags.DEFINE_integer(
    'num_iter', 3,
    'Number of iterative refinement steps')
flags.DEFINE_integer(
    'd_slot', 128,
    'Dimension of the layers in the attn module')
flags.DEFINE_integer(
    'd_mlp', 128,
    'Dimension of the layers in the MLP head')
flags.DEFINE_float(
    'eps', 1e-08,
    'Offset for attn coeffs before normalization')
flags.DEFINE_bool(
    'use_implicit', True,
    'Set whether to train slot attn using implicit differentiation')
flags.DEFINE_string(
    'alpha_mask_type', 'linear',
    'Set which activation function to use to normalize alpha mask. softmax, sigmoid (default), linear. Set to linear to use no alpha mask')
flags.DEFINE_string(
    'encoder_type', 'simple',
    'encoder backbone, simple (Default), resnet18, resnet34')
flags.DEFINE_bool(
    'share_slot_init', False,
    'whether to share the initialization of slots across all slots')
flags.DEFINE_integer(
    'num_encoder_layers', 4,
    'Number of layers in the SimpleEncoder. Default: 4')
flags.DEFINE_integer(
    'kernel_height', 5,
    'Height of convolution kernal in the encoder layers. Default: 5')
flags.DEFINE_integer(
    'num_strided_decoder_layers', 4,
    'Number of strided transposed convolution layers in the decoder. Default: 4')
flags.DEFINE_bool(
    'use_deconv', True,
    'whether to use deconv/transposed conv (True by default) or upsample + conv')
flags.DEFINE_integer(
    'freq_stride', 1,
    'Stride length across frequency. Default: 1')
flags.DEFINE_integer(
    'time_stride', 1,
    'Stride length across time. Default: 1')

# Wandb args
flags.DEFINE_bool(
    'log_wandb', True,
    'Set whether to log results on Wandb.ai')
flags.DEFINE_string(
    'project', 'jazznet_multi-instrument',
    'Name of the project')
flags.DEFINE_string(
    'name', None,
    'Name of the run')
flags.DEFINE_integer(
    'num_to_log', 3,
    'Set how many example plots to log on Wandb.ai')
flags.DEFINE_integer(
    'logging_epoch_freq', 25,
    'Set how frequently the plots should be logged on Wandb.ai, every 25th epoch by default')

FLAGS = flags.FLAGS

def main(argv):
    del argv

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.spec == 'cqt':
        db_thres = -45
    elif FLAGS.spec == 'mel':
        db_thres = -30

    if FLAGS.dataset == 'jazznet':
        root = 'data/jazznet_single'

    elif FLAGS.dataset == 'jazznet_multi_inst':
        root = 'data/jazznet_multi'

    elif FLAGS.dataset == 'bach_chorales':
        root = 'data/jsb_single'

    elif FLAGS.dataset == 'bach_chorales_multi_inst':
        root = 'data/jsb_multi'

    dm = MusicalObjectDataModule(
        root = root,
        to_db = FLAGS.to_db,
        spec = FLAGS.spec,
        top_db = FLAGS.top_db,
        batch_size = FLAGS.batch_size,
        num_workers = FLAGS.num_workers,
        seed = FLAGS.seed
    )
    
    img_transforms = [transforms.Lambda(
        partial(spec_crop, height=int(FLAGS.img_size[0]), width=int(FLAGS.img_size[1])))]

    train_transforms = transforms.Compose(img_transforms)
    test_transforms = transforms.Compose(img_transforms)

    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms

    model = SlotAttentionAE(
        in_channels = 1,
        resolution = [int(FLAGS.img_size[0]), int(FLAGS.img_size[1])],
        lr = FLAGS.lr,
        warmup_steps = FLAGS.warmup_steps,
        decay_steps = FLAGS.decay_steps,
        num_slots = FLAGS.num_slots,
        num_iter = FLAGS.num_iter,
        d_slot = FLAGS.d_slot,
        d_mlp = FLAGS.d_mlp,
        eps = FLAGS.eps,
        use_implicit = FLAGS.use_implicit,
        share_slot_init = FLAGS.share_slot_init,
        alpha_mask_type = FLAGS.alpha_mask_type,
        kernel_height = FLAGS.kernel_height,
        num_encoder_layers = FLAGS.num_encoder_layers,
        num_strided_decoder_layers = FLAGS.num_strided_decoder_layers,
        use_deconv = FLAGS.use_deconv,
        grad_clip_val = FLAGS.grad_clip_val,
        stride = (FLAGS.freq_stride, FLAGS.time_stride),
        db_thres = db_thres,
        to_db = FLAGS.to_db,
        top_db = FLAGS.top_db,
        encoder_type = FLAGS.encoder_type,
        num_to_log = FLAGS.num_to_log,
        logging_epoch_freq = FLAGS.logging_epoch_freq
    )

    if FLAGS.log_wandb:
        if FLAGS.name is None:
            name = 'sweep_run_' + 'slots={}_'.format(FLAGS.num_slots) + 'iter={}_'.format(FLAGS.num_iter) + str(FLAGS.alpha_mask_type)
        else:
            name = FLAGS.name
        save_dir = '/data/joonsu' if os.path.exists('/data/joonsu') else '.'
        wandb_logger = WandbLogger(
            project=FLAGS.project, name=name, save_dir=save_dir, log_model=True)
        wandb_logger.watch(model) # log gradients and model topology
    else:
        wandb_logger = None

    if FLAGS.use_gpu:
        accelerator = "gpu"
        find_unused_parameters = True if FLAGS.use_implicit else False
        if str(-1) in FLAGS.device_id:
            devices = -1
            strategy = DDPStrategy(
                find_unused_parameters=find_unused_parameters)
        else:
            devices = [int(i) for i in FLAGS.device_id]
            if len(devices) == 1:
                strategy = "auto"
            else:
                strategy = DDPStrategy(
                    find_unused_parameters=find_unused_parameters)
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    cb = [TQDMProgressBar(refresh_rate=10)]
    model_ckpt = ModelCheckpoint(monitor="val_best_ov_iou", mode="max")
    cb.append(model_ckpt)
    
    if FLAGS.log_wandb:
        cb.append(LearningRateMonitor(logging_interval="step"))

    if FLAGS.use_mixed_precision:
        precision = '16-mixed'
    else:
        precision = 32

    # max_steps needs to be multiplied by the number of optimizers since max_steps = total optimizer steps...
    trainer = Trainer(
        max_steps=FLAGS.max_steps * len(model.configure_optimizers()[0]),
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        strategy=strategy,
        callbacks=cb,
        precision=precision)

    trainer.fit(model, dm)
    trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)

    # Evaluate the learned representation on downstream task
    backbone = model.load_from_checkpoint(model_ckpt.best_model_path)
    classifier = SlotClassifier(
        backbone,
        FLAGS.classifier_lr,
        dm.num_notes,
        dm.num_instruments,
        linear=FLAGS.use_linear_classifier)

    classifier_cb = [TQDMProgressBar(refresh_rate=10)]
    classifier_ckpt = ModelCheckpoint(monitor="val_acc", mode="max")
    classifier_cb.append(classifier_ckpt)

    trainer = Trainer(
        max_steps=FLAGS.classifier_steps,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        strategy=strategy,
        callbacks=classifier_cb,
        precision='16-mixed' if FLAGS.use_gpu else 32)
    
    trainer.fit(classifier, dm)
    trainer.test(classifier, datamodule=dm)

if __name__ == '__main__':
    app.run(main)