from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
try:
    from dataset.utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from dataset.definitions import id_map
except:
    from utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from definitions import id_map, custom_colormap


class SemanticTHAB(Dataset):
    def __init__(self, data_path, rotate=False, flip=False, id_map=id_map, projection=None, resize=False):
        self.id_map = id_map
        self.data_path = data_path
        self.rotate = rotate
        self.flip = flip
        self.projection=projection
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


def main(path_to_dataset, split_type, num_classes, visu=False):
    import glob
    import cv2 
    import tqdm
    if split_type=="train":
        data_path = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:04}" for i in range(9) if i != 6] for bin_path in glob.glob(f"{path_to_dataset}/{folder}/velodyne/*.bin")]
    elif split_type=="test":
        data_path = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{path_to_dataset}/0006/velodyne/*.bin")] 
    else:
        raise ValueError
    
    dataset= SemanticTHAB(data_path, rotate=False, flip=False, projection=(128,2048),resize=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1 if visu==True else 16)

    total_counts = np.zeros(num_classes, dtype=np.int64)
    for batch_idx, (range_img, reflectivity, xyz, normals, semantic) in enumerate(tqdm.tqdm(dataloader)):
        if visu:
            semantics = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...]#.cpu().detach().numpy()
        else:
            semantics = semantic.squeeze(1).numpy()
        # number of points per class statistics
        counts = np.bincount(semantics.reshape(-1).astype(np.int64), minlength=num_classes)
        total_counts += counts
        
        #if batch_idx > 20: break
        if visu:
            reflectivity = (reflectivity[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
            normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
            prev_sem_pred = cv2.applyColorMap(np.uint8(semantics), custom_colormap)
            cv2.imshow("semseg", prev_sem_pred[...,::-1])
            cv2.imshow("reflectivity", cv2.applyColorMap(np.uint8(255*reflectivity),cv2.COLORMAP_JET))
            cv2.waitKey(0)
    
    class_counts = {i: total_counts[i] for i in range(num_classes)}
    from definitions import class_names, color_map
    from utils import plot_pointCounts_per_class
    
    plot_pointCounts_per_class(
        class_counts,
        class_names=list(class_names.values()),
        num_classes=num_classes,
        color_map=color_map,
        ignore_ids=(0,),          # ignore 'unlabeled'
        log_scale=True,
        annot_rotation=50,         # horizontal totals
        annot_offset=5,          # push labels a bit higher    
        top_pad=None,               # even more headroom if needed , None for auto adjust top height
        title=f"SemanticTHAB Class Distribution [{split_type} split]",
        save_path=f"/home/devuser/workspace/src/dataset/class_distributions/classDistribution_SemanticTHAB_{split_type}.png"
    )
    print(class_counts)
    print("END")

if __name__ == "__main__":
    path_to_dataset = "/home/devuser/workspace/data/semantic_datasets/SemanticTHAB/sequences"
    split_type = "train" # choose dataset split, options: "train" | "test"
    num_classes = 20
    main(path_to_dataset, split_type, num_classes, visu=False)