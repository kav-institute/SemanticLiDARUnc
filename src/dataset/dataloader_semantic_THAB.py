from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from dataset.utils import rotate_equirectangular_image, rotate_z, build_normal_xyz
from dataset.definitions import id_map



class SemanticKitti(Dataset):
    def __init__(self, data_path, rotate=False, flip=False, id_map=id_map):
        self.id_map = id_map
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
        
        #cls_dict = {"Car": 1, "Bus": 2, "Truck": 3, "Pedestrian": 4, "Bicyclist": 5, "Motorcyclist": 6, "TrafficCone": 7}
        
        
        frame_path, label_path = self.data_path[idx]
        # the (x, y, z, intensity) are stored in binary
        xyzi = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4) 

        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        
        # get semantic labels
        sem_label = label & 0xFFFF

        #get instance labels
        inst_label = label >> 16
        
        # map semantic labels
        #sem_label_map = np.array([id_map[l] for l in sem_label])id_map_dynamic
        sem_label_map = np.array([id_map[l] for l in sem_label])
        sem_label_map = sem_label_map.reshape((128,2048,1))
        xyzi = xyzi.reshape((128,2048,4))
        xyzi_img = np.concatenate((xyzi,sem_label_map),axis=-1)
        if self.flip:
            if np.random.choice([True, False]):
                xyzi_img = xyzi_img[:,::-1,:]
                xyzi_img[...,1] = -xyzi_img[...,1]
        if self.rotate:
            random_angle = np.random.randint(-180,180)
            xyzi_img = rotate_equirectangular_image(xyzi_img, random_angle)
            xyzi_img[...,0:3] = rotate_z(xyzi_img[...,0:3].reshape(-1,3), random_angle).reshape(xyzi_img[...,0:3].shape)
            #xyzi_img = np.rollaxis(xyzi_img,1,random_angle)
        #xyzi_img = xyzi_img[100:200,:,:]
        #xyzi_img = xyzi_img[~np.all(np.linalg.norm(xyzi_img,axis=-1) == 0, axis=1)]
        label_img = xyzi_img[...,4:5]
        reflectivity_img = xyzi_img[...,3]
        xyzi_img = xyzi_img[...,0:3]
        
        range_img = np.linalg.norm(xyzi_img,axis=-1)
        
        normals = build_normal_xyz(xyzi_img[...,0:3])
        
        label_img = label_img

        semantics =  label_img

        reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))
        xyz =  torch.as_tensor(xyzi_img[...,0:3].transpose(2, 0, 1).astype("float32"))
        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))

        semantics =  torch.as_tensor(semantics.transpose(2, 0, 1).astype("long"))
        
        # xyz_tensor = torch.as_tensor(xyzi[...,0:3].astype("float32"))
        # semantics_tensor = torch.as_tensor(sem_label_map[...,None].astype("float32"))
        return range_img, reflectivity_img, xyz, normals, semantics
