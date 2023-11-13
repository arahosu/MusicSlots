from absl import app
from absl import flags

import os
import gdown

flags.DEFINE_enum(
    'dataset', 'jsb_single',
    ['jsb_single', 'jsb_multi', 'jazznet_single', 'jazznet_multi'],
    'dataset to be downloaded, jsb_single (default), jsb_multi, jazznet_single or jazznet_multi')
flags.DEFINE_string(
    'savedir', None,
    'directory where the file is saved')

flags.mark_flag_as_required('savedir')

FLAGS = flags.FLAGS

def main(_):
    if FLAGS.dataset == 'jsb_single':
        url = 'https://drive.google.com/u/0/uc?id=1yOGSvCXOGj4lu1ydc_G4mJMIchZGdxKO&export=download'
    elif FLAGS.dataset == 'jsb_multi':
        url = 'https://drive.google.com/u/0/uc?id=1VzDAJWH4WY025_9L-Z1u3sJQGgZtmRsS&export=download'
    elif FLAGS.dataset == 'jazznet_single':
        url = 'https://drive.google.com/u/0/uc?id=1lgtTk3ZEtiDU961cH0LZlEVzKC_OkTuH&export=download'
    elif FLAGS.dataset == 'jazznet_multi': 
        url = 'https://drive.google.com/u/0/uc?id=1WqfrNxCruy8xharqZT6_zeDzkKQ5d1Oq&export=download'

    zip_file = FLAGS.dataset + '.zip'
    output = os.path.join(FLAGS.savedir, zip_file)

    # Download file
    gdown.download(url, output)

if __name__ == '__main__':
    app.run(main)