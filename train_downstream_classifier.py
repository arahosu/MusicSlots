import torch
from torchvision import transforms

from absl import app
from absl import flags
import os
import random
from functools import partial
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from dataset import MusicalObjectDataModule, spec_crop

from models.Baselines.vae import VanillaVAE, VAE_Classifier
from models.MusicSlots.slot_attention import SlotAttentionAE
from models.MusicSlots.classifier import SlotClassifier

# Random seed
flags.DEFINE_integer(
    'seed', 42,
    'random seed')

# Dataset flags
flags.DEFINE_string(
    'dataset', 'jazznet_multi_inst',
    'bach_chorales, jazznet, jazznet_multi_inst')
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

# Downstream training flags
flags.DEFINE_string(
    'model_dir', None,
    'directory where the model is saved')
flags.DEFINE_string(
    'task', 'note',
    'Downstream prediction task. note (default) or chord')
flags.DEFINE_integer(
    'num_workers', 8,
    'set the number of workers for the dataloader')
flags.DEFINE_integer(
    'batch_size', 32,
    'set batch size for evaluation.')
flags.DEFINE_integer(
    'max_steps', 10000,
    'maximum number of training steps')
flags.DEFINE_float(
    'classifier_lr', 1e-03,
    'set learning rate for classifiers. 1e-03 (default)')
flags.DEFINE_bool(
    'use_linear_classifier', True,
    'whether to train a downstream linear classifier (Default) or MLP classifier')
flags.DEFINE_integer(
    'test_time_iter', 3,
    'Number of iterative refinement steps at test time. Default: 3')

# Wandb args
flags.DEFINE_bool(
    'log_wandb', True,
    'Set whether to log results on Wandb.ai')
flags.DEFINE_string(
    'project', 'mel_sweep_3',
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
        root = './data/jazznet_single'

    elif FLAGS.dataset == 'jazznet_multi_inst':
        root = './data/jazznet_multi'

    elif FLAGS.dataset == 'bach_chorales':
        root = './data/jsb_single'

    elif FLAGS.dataset == 'bach_chorales_multi_inst':
        root = './data/jsb_multi'

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
    
    ckpt = torch.load(FLAGS.model_dir)

    if 'kld_weight' in ckpt['hyper_parameters']:
        backbone = VanillaVAE.load_from_checkpoint(FLAGS.model_dir)
        model = VAE_Classifier(
            backbone, FLAGS.classifier_lr,
            dm.num_notes, dm.num_instruments, FLAGS.use_linear_classifier)
    else:
        backbone = SlotAttentionAE.load_from_checkpoint(FLAGS.model_dir, strict=True, num_iter=FLAGS.test_time_iter)
        model = SlotClassifier(
            backbone, FLAGS.classifier_lr,
            dm.num_notes, dm.num_instruments, FLAGS.use_linear_classifier)

    if FLAGS.log_wandb:
        if FLAGS.name is None:
            name = 'downstream_classifier_run'
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

    trainer = Trainer(
        max_steps=FLAGS.max_steps,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        strategy=strategy,
        callbacks=cb,
        precision='16-mixed' if FLAGS.use_gpu else 32)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
        

if __name__ == '__main__':
    app.run(main)