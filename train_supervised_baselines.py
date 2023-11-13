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
from models.Baselines.supervised import SupervisedClassifier

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
    'whether to convert the amplitude of the spectrograms to db scale. False (default)')
flags.DEFINE_float(
    'top_db', 80.0,
    'threshold the output at top_db below the peak. 80.0 (default)')

# Accelerator flags
flags.DEFINE_bool(
    'use_gpu', True,
    'set whether to use GPU')
flags.DEFINE_list(
    'device_id', [0], 
    'set which GPU/TPU device to use')

# Training flags
flags.DEFINE_integer(
    'num_workers', 8,
    'set the number of workers for the dataloader')
flags.DEFINE_integer(
    'batch_size', 32,
    'set batch size for training.')
flags.DEFINE_float(
    'lr', 1e-03,
    'set learning rate for training')
flags.DEFINE_float(
    'grad_clip_val', 0.1,
    'set value for gradient clipping')
flags.DEFINE_integer(
    'max_steps', 10000,
    'Maximum number of steps for training')

# Model args
flags.DEFINE_string(
    'backbone', 'simple',
    'Backbone network for the classifier. simple (default), resnet or vgg')
flags.DEFINE_float(
    'dropout_ratio', 0.5,
    'Dropout ratio for VGG network')
flags.DEFINE_integer(
    'freq_stride', 2,
    'ResNet stride across frequency. 2 (default)')
flags.DEFINE_integer(
    'time_stride', 2,
    'ResNet stride across time. 2 (default)')

# Wandb args
flags.DEFINE_bool(
    'log_wandb', True,
    'Set whether to log results on Wandb.ai')
flags.DEFINE_string(
    'project', 'baselines',
    'Name of the project')
flags.DEFINE_string(
    'name', None,
    'Name of the run')

FLAGS = flags.FLAGS

def main(argv):
    del argv

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

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
        partial(spec_crop, height=FLAGS.img_size[0], width=FLAGS.img_size[1]))]

    train_transforms = transforms.Compose(img_transforms)
    test_transforms = transforms.Compose(img_transforms)

    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms

    model = SupervisedClassifier(in_channels=1,
                                 lr=FLAGS.lr,
                                 resolution=FLAGS.img_size,
                                 backbone=FLAGS.backbone,
                                 num_notes=dm.num_notes,
                                 num_instruments=dm.num_instruments,
                                 stride=(FLAGS.freq_stride, FLAGS.time_stride))

    if FLAGS.log_wandb:
        if FLAGS.name is None:
            name = 'supervised_' + FLAGS.backbone
        else:
            name = FLAGS.name
        save_dir = '/data/joonsu' if os.path.exists('/data/joonsu') else '.'
        wandb_logger = WandbLogger(project=FLAGS.project, name=name, save_dir=save_dir)
    else:
        wandb_logger = None

    if FLAGS.use_gpu:
        accelerator = "gpu"
        if str(-1) in FLAGS.device_id:
            devices = -1
            strategy = DDPStrategy(
                find_unused_parameters=False)
        else:
            devices = [int(i) for i in FLAGS.device_id]
            if len(devices) == 1:
                strategy = "auto"
            else:
                strategy = DDPStrategy(
                    find_unused_parameters=False)
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    cb = [TQDMProgressBar(refresh_rate=10)]
    model_ckpt = ModelCheckpoint(monitor="val_chord_acc", mode="max")
    cb.append(model_ckpt)
    
    if FLAGS.log_wandb:
        cb.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        max_steps=FLAGS.max_steps,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        strategy=strategy,
        callbacks=cb,
        precision='16-mixed' if FLAGS.use_gpu else 32)

    trainer.fit(model, dm)
    trainer.test(model.load_from_checkpoint(model_ckpt.best_model_path), datamodule=dm)

if __name__ == '__main__':
    app.run(main)