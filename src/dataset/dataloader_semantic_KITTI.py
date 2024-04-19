from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from dataset.utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
from dataset.definitions import id_map
import cv2



class SemanticKitti(Dataset):
    def __init__(self, data_path, rotate=False, flip=False):
        self.data_path = data_path
        self.rotate = rotate
        self.flip = flip
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # Add more transformations if needed
        ])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        frame_path, label_path = self.data_path[idx]
        # the (x, y, z, intensity) are stored in binary
        xyzi = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4) 

        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        
        # get semantic labels
        sem_label = label & 0xFFFF

        #get instance labels
        inst_label = label >> 16
        

        sem_label_map = np.array([id_map[l] for l in sem_label])

        xyzil = np.concatenate([xyzi, sem_label_map[...,np.newaxis]],axis=-1)

        # Augmentations
        
        if self.rotate:
            random_angle = np.random.randint(-180,180)
            xyzil[...,0:3] = rotate_z(xyzil[...,0:3].reshape(-1,3), random_angle)
        xyzi_img, _, _ , _ = spherical_projection(xyzil,64,2048)
        xyzi_img = cv2.resize(xyzi_img, (2048,128), interpolation=cv2.INTER_NEAREST)
        
        if self.flip:
            do_flip = np.random.choice([True, False])
            if do_flip:
                xyzi_img = xyzi_img[:,::-1,:]
                xyzi_img[...,1] = -xyzi_img[...,1]

        label_img = xyzi_img[...,4:5]
        reflectivity_img = xyzi_img[...,3]
        xyzi_img = xyzi_img[...,0:3]
        range_img = np.linalg.norm(xyzi_img,axis=-1)
        
        normals = build_normal_xyz(xyzi_img[...,0:3])
        
        label_img = label_img

        semantics =  label_img

        reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))
        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        xyz =  torch.as_tensor(xyzi_img[...,0:3].transpose(2, 0, 1).astype("float32"))

        normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))

        semantics =  torch.as_tensor(semantics.transpose(2, 0, 1).astype("long"))
        
        return range_img, reflectivity_img, xyz, normals, semantics