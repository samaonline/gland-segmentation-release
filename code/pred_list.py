import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import os
import sys
import argparse
import numpy as np
from PIL import Image

from pdb import set_trace as st


parser = argparse.ArgumentParser()
parser.add_argument('--lst', required=True, help='Text file that contains image paths')
parser.add_argument('--model', required=True, help='Path of the model')
parser.add_argument('--usegpu', action='store_true',
                    help='Enables cuda to predict on gpu')
parser.add_argument('--dataset', type=str,
                    help='Name of the dataset which is "CVPPP"',
                    required=True)
opt = parser.parse_args()

assert opt.dataset in ['CVPPP', 'meibo']

images_list = np.loadtxt(opt.lst, dtype='str', delimiter=',')
model_path = opt.model

_subset = os.path.basename(opt.lst).split('_')[0]
_model_name = os.path.splitext(os.path.basename(model_path))[0]
_model_dir = os.path.basename(os.path.dirname(model_path))

# output_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir,
#                                            os.path.pardir, 'outputs', opt.dataset,
#                                            _model_dir + '-' + _model_name, _subset))

sub_output_path = os.path.join(opt.dataset, _model_dir + '-' + _model_name, _subset)
output_path = "/home/peterwg/repos/instance-segmentation-pytorch/outputs/{}".format(sub_output_path)

image_names = [os.path.splitext(os.path.basename(img))[0] \
    for img in images_list]

try:
    os.makedirs(output_path)
except BaseException:
    pass

model_dir = os.path.dirname(model_path)
sys.path.insert(0, model_dir)

from lib import Model, Prediction

if opt.dataset == 'CVPPP':
    from settings import CVPPPModelSettings
    ms = CVPPPModelSettings()
elif opt.dataset == 'meibo':
    from settings import MeiboModelSettings
    ms = MeiboModelSettings()


model = Model(opt.dataset, ms.MODEL_NAME, ms.N_CLASSES, ms.MAX_N_OBJECTS,
              use_instance_segmentation=ms.USE_INSTANCE_SEGMENTATION,
              use_coords=ms.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

prediction = Prediction(ms.IMAGE_HEIGHT, ms.IMAGE_WIDTH,
                        ms.MEAN, ms.STD, False, model,
                        1)

for image_name, image_path in zip(image_names, images_list):

    _output_path = os.path.join(output_path, image_name)
    
    try:
        os.makedirs(_output_path)
    except BaseException:
        pass

    image, fg_seg_pred, ins_seg_pred, n_objects_pred = prediction.predict(image_path) #, _output_path)
    
    #############################################
    
#     print("=========== pred_list ============")
    
#     print(type(image))
#     print(image.shape)
    
#     print(type(fg_seg_pred))
#     print(fg_seg_pred.shape)
    
#     print(type(ins_seg_pred))
#     print(ins_seg_pred.shape)
    
#     print(n_objects_pred)
    
    #############################################

    fg_seg_pred = fg_seg_pred * 255

    _n_clusters = len(np.unique(ins_seg_pred.flatten())) - 1  # discard bg
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
    ins_seg_pred_color = np.zeros(
        (ins_seg_pred.shape[0], ins_seg_pred.shape[1], 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (
            i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

    image_pil = Image.fromarray(image)
    fg_seg_pred_pil = Image.fromarray(fg_seg_pred)
    ins_seg_pred_pil = Image.fromarray(ins_seg_pred)
    ins_seg_pred_color_pil = Image.fromarray(ins_seg_pred_color)

    image_pil.save(os.path.join(_output_path, image_name + '.png'))
    fg_seg_pred_pil.save(os.path.join(_output_path, image_name + '-fg_mask.png'))
    ins_seg_pred_pil.save(os.path.join(_output_path, image_name + '-ins_mask.png'))
    ins_seg_pred_color_pil.save(os.path.join(
        _output_path, image_name + '-ins_mask_color.png'))
    np.save(
        os.path.join(
            _output_path,
            image_name +
            '-n_objects.npy'),
        n_objects_pred)
