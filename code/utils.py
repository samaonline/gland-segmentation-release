from os.path import join
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from lib.utils import ImageUtilities as IU
from PIL import Image
import random
from lib import AlignCollate


class H5Dataset(Dataset):
    """Dataset Reader"""

    def __init__(self, h5_path):

        self._h5_path = h5_path
        
        data = h5py.File(join(self._h5_path, "data.h5") , 'r')
        try:
            self.images = np.array(data['images'])
            self.semantics = np.array(data['semantics'])
            self.infos = np.array(data['infos'])
            data.close()
            
            with open(join(self._h5_path, "instances.pkl"), "rb") as file:
                self.instances = pickle.load(file)
            
            self.n_samples = self.images.shape[0]
        except Exception as e:
            data.close()
            raise RuntimeError("Error triggered [utils.load_data]: {}".format(str(e)))

    def __load_data(self, index):
        
        image = self.images[index]
        image = torchvision.transforms.ToPILImage(mode=None)(image)
        
        semantic = self.semantics[index]
        instance = self.instances[index]
        n_objects = self.infos[index][-1]
        
        return image, semantic, instance, n_objects

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        image, semantic, instance, n_objects = self.__load_data(index)

        return image, semantic, instance, n_objects

    def __len__(self):
        return self.n_samples

    
if __name__ == '__main__':
    ds = H5Dataset('../data/processed/meibo/h5/training-h5/')
    
    image, semantic_annotation, instance_annotation, n_objects = ds[5]

    print("============== MAIN 1 ================")
    print(image.size)
    print(semantic_annotation.shape)
    print(instance_annotation.shape)
    print(n_objects)
    print(np.unique(semantic_annotation))
    print(np.unique(instance_annotation))

    ac = AlignCollate('training', 9, 120, [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0], 256, 512)

    loader = torch.utils.data.DataLoader(ds, batch_size=3,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=False,
                                         collate_fn=ac)
    loader = iter(loader)

    images, semantic_annotations, instance_annotations, \
        n_objects = next(loader)

    print("============== MAIN 2 ================")
    print(images.size())
    print(semantic_annotations.size())
    print(instance_annotations.size())
    print(n_objects.size())
    print(n_objects)