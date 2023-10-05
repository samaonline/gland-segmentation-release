import os
import glob
import cv2
from PIL import Image
import numpy as np
import h5py
from pdb import set_trace as st
from tqdm import *

datasets = 'meibo' #CVPPP

data_path = "/home/peterw/repo/instance-segmentation-pytorch/data/dataset/meibo/save_img/labeled_data/LL/train/glands"
raw_dir = "/home/peterw/repo/instance-segmentation-pytorch/data/dataset/meibo/raw_img/LL/train"

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,                                        os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'raw', datasets, datasets+'2017_LSC_training',
                       'training', 'A1')
IMG_DIR = os.path.join(DATA_DIR, 'raw', datasets, datasets+'2017_LSC_training',
                       'training', 'A1')

subfolders = [ f.path for f in os.scandir(data_path) if f.is_dir() ]

SEMANTIC_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', datasets,
                                   'semantic-annotations')
INSTANCE_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', datasets,
                                   'instance-annotations')
#INSTANCE_OUTPUT_DIR = "/home/peterw/temp"

try:
    os.makedirs(SEMANTIC_OUTPUT_DIR)
except BaseException:
    pass

'''try:
    os.makedirs(INSTANCE_OUTPUT_DIR)
except BaseException:
    pass'''

image_paths = glob.glob(os.path.join(IMG_DIR, '*_rgb.png'))

output = {}
output['images'] = []
output['semantics'] = []
output['instances'] = []
output['n_objects'] = []
output['names'] = []

for image_path in tqdm(subfolders): #[:380]): #image_paths:
    identifier =  image_path.split('/')[-1]
    num, whichone = identifier.split("_")

    raw_path = os.path.join( raw_dir, whichone+num+".png")
    img = cv2.imread(raw_path)

    filelist = [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.png')]
    filelist.sort()

    instance_mask = [(cv2.imread(i, cv2.IMREAD_GRAYSCALE) !=0).astype(int) for i in filelist]
    instance_mask = np.array(instance_mask)
    instance_mask = np.transpose(instance_mask, (1,2,0))

    semantic_mask = instance_mask.sum(2)
    semantic_mask[semantic_mask != 0] = 1
    semantic_mask = semantic_mask.astype(np.uint8)

    output['images'].append(img)
    output['semantics'].append(semantic_mask)
    output['instances'].append(instance_mask)
    output['n_objects'].append(instance_mask.shape[-1])
    #output['names'].append(identifier)

np.save("train", output)

"""if False:
    img = Image.open(raw_path)
    img_width, img_height = img.size

    image_name = os.path.splitext(os.path.basename(image_path))[
        0].split('_rgb')[0]

    # print(os.path.splitext(os.path.basename(image_path))[0])
    # exit(0)
    
    annotation_path = os.path.join(ANN_DIR, image_name + '_label.png')


    if not os.path.isfile(annotation_path):
        continue

    anno_img = Image.open(annotation_path)
    #anno_img = anno_img.convert('L')
    #anno_img.save('{}.png'.format(image_name))

    annotation = np.array(anno_img)


    assert len(annotation.shape) == 2
    assert np.array(img).shape[:2] == annotation.shape[:2]

    instance_values = set(np.unique(annotation)).difference([0])


    # # Define a dictionary to map instance values to labels
    # instance_mapping = {
    #     225: 1,
    #     104: 2,
    #     105: 2,
    #     75: 3,
    #     76: 3,
    #     178: 4,
    #     149: 5,
    #     28: 6,
    #     29: 6,
    #     254: 7
    # }

    # # Transform instance values to labels using the dictionary
    # for instance_value, label in instance_mapping.items():
    #     annotation[annotation == instance_value] = label


    # if 
    # print(instance_values)
    #exit()

    n_instances = len(instance_values)



    if n_instances == 0:
        continue

    instance_mask = np.zeros(
        (img_height, img_width, n_instances), dtype=np.uint8)

    for i, v in enumerate(instance_values):
        _mask = np.zeros((img_height, img_width), dtype=np.uint8)
        _mask[annotation == v] = 1
        instance_mask[:, :, i] = _mask

    semantic_mask = instance_mask.sum(2)
    semantic_mask[semantic_mask != 0] = 1
    semantic_mask = semantic_mask.astype(np.uint8)

    np.save(os.path.join(INSTANCE_OUTPUT_DIR, image_name + '.npy'),
            instance_mask)
    np.save(os.path.join(SEMANTIC_OUTPUT_DIR, image_name + '.npy'),
            semantic_mask)"""
