import argparse
import random
import os
import getpass
import datetime
import shutil
import numpy as np
import torch
from lib import NPYDataset, SegDataset, Model, AlignCollate


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='',
                    help="Filepath of trained model (to continue training) \
                         [Default: '']")
parser.add_argument('--usegpu', type=bool, default=True,
                    help='Enables cuda to train on gpu [Default: True]')
parser.add_argument('--nepochs', type=int, default=300,
                    help='Number of epochs to train for [Default: 600]')
parser.add_argument('--batchsize', type=int,
                    default=8, help='Batch size [Default: 2]')
parser.add_argument('--debug', action='store_true',
                    help='Activates debug mode [Default: False]')
parser.add_argument('--nworkers', type=int,
                    help='Number of workers for data loading \
                        (0 to do it using main process) [Default : 2]',
                    default=2)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--dataset', type=str,
                    help='Name of the dataset which is "CVPPP"',
                    required=True)
parser.add_argument('--data_folder', type=str, default='h5-amodal-standard',
                    help='Which H5 data to use for meibo',
                    required=False)
            

opt = parser.parse_args()

assert opt.dataset in ['CVPPP', 'meibo']

if opt.dataset == 'CVPPP':
    from settings import CVPPPTrainingSettings
    ts = CVPPPTrainingSettings()
elif opt.dataset == 'meibo':
    from settings import MeiboTrainingSettings
    ts = MeiboTrainingSettings(opt.data_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = list(map(str, [now.year, now.month, now.day]))
    coarse_time = list(map(str, [now.hour, now.minute]))
    fine_time = list(map(str, [now.second, now.microsecond]))

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time),
                       username, '-'.join(fine_time)])
    return run_id


RUN_ID = generate_run_id()

# model_save_path = os.path.abspath(os.path.join(os.path.abspath(__file__),
#                                                os.path.pardir, os.path.pardir,
#                                                'models', opt.dataset, RUN_ID))
model_save_path = "../models/{}".format(os.path.join(opt.dataset, RUN_ID))

os.makedirs(model_save_path)

CODE_BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), os.path.pardir))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'settings'),
                os.path.join(model_save_path, 'settings'))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'lib'),
                os.path.join(model_save_path, 'lib'))

if torch.cuda.is_available() and not opt.usegpu:
    print('WARNING: You have a CUDA device, so you should probably \
        run with --usegpu')

# Load Seeds
random.seed(ts.SEED)
np.random.seed(ts.SEED)
torch.manual_seed(ts.SEED)

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

if opt.dataset == "CVPPP":
    train_dataset = SegDataset(ts.TRAINING_LMDB)
elif opt.dataset == "meibo":
    train_dataset = NPYDataset("/home/peterwg/repos/instance-segmentation-pytorch/data/train.npy", "train") #H5Dataset(ts.TRAINING_H5, "train")
assert train_dataset

train_align_collate = AlignCollate(
    'training',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH,
    random_hor_flipping=ts.HORIZONTAL_FLIPPING, 
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING
)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=opt.batchsize,
                                           shuffle=True,
                                           num_workers=opt.nworkers,
                                           pin_memory=pin_memory,
                                           collate_fn=train_align_collate)

if opt.dataset == "CVPPP":
    test_dataset = SegDataset(ts.TRAINING_LMDB)
elif opt.dataset == "meibo":
    test_dataset = NPYDataset("/home/peterwg/repos/instance-segmentation-pytorch/data/test.npy", "test") #H5Dataset(ts.TRAINING_H5, "test")
assert test_dataset

test_align_collate = AlignCollate(
    'test',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH,
    random_hor_flipping=ts.HORIZONTAL_FLIPPING,
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=opt.batchsize,
                                          shuffle=False,
                                          num_workers=opt.nworkers,
                                          pin_memory=pin_memory,
                                          collate_fn=test_align_collate)

# Define Model
model = Model(opt.dataset, ts.MODEL_NAME, ts.N_CLASSES, ts.MAX_N_OBJECTS,
              use_instance_segmentation=ts.USE_INSTANCE_SEGMENTATION,
              use_coords=ts.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

# Train Model
model.fit(ts.CRITERION, ts.DELTA_VAR, ts.DELTA_DIST, ts.NORM, ts.LEARNING_RATE,
          ts.WEIGHT_DECAY, ts.CLIP_GRAD_NORM, ts.LR_DROP_FACTOR,
          ts.LR_DROP_PATIENCE, ts.OPTIMIZE_BG, ts.OPTIMIZER, ts.TRAIN_CNN,
          opt.nepochs, ts.CLASS_WEIGHTS, train_loader, test_loader,
          model_save_path, opt.debug)
