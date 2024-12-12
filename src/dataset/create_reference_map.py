import numpy as np
import open3d as o3d

from vdbfusion import VDBVolume
import numpy as np
from tqdm import tqdm
import os
from trimesh import transform_points
import pandas as pd
import glob
import cv2
import cc3d
#import matplotlib.pyplot as plt

class EquirectangularRayCasting:
    def __init__(self, height=256, origin=(0,0,0)):
        
        self.origin = origin
        self.height = height
        width = 2*height
        self.width = width
        phi_min, phi_max = [-np.pi, np.pi]
        theta_min, theta_max = [-np.pi/2, np.pi/2]
        # assuming uniform distribution of rays
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
        bins_w = np.linspace(phi_min, phi_max, width)[::-1]
        
        theta_img = np.stack(width*[bins_h], axis=-1)
        phi_img = np.stack(height*[bins_w], axis=0)

        
        x = np.sin(theta_img+np.pi/2)*np.cos(phi_img)#+np.pi)
        y = np.sin(theta_img+np.pi/2)*np.sin(phi_img)#+np.pi)
        z = -np.cos(theta_img+np.pi/2)

        self.ray_img = np.stack([x,y,z],axis=-1)
        self.origin_img_ = np.ones_like(self.ray_img)

        self.merged_mesh = o3d.geometry.TriangleMesh()
        self.createScene()

        

    def createScene(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        #self.scene = o3d.geometry.RaycastingScene()
    
    def get_colors(self, mesh, primitive_ids):
        valid_mask = np.where(primitive_ids != self.scene.INVALID_ID, 1, 0)
        primitive_ids_masked = valid_mask*primitive_ids
        vertex_indices = np.asarray(mesh.triangles)[primitive_ids_masked.flatten()][...,0]
        colors = np.asarray(mesh.vertex_colors)[vertex_indices]
        colors = colors.reshape(primitive_ids.shape + (3,))
        colors = np.where(valid_mask[...,None], colors, [0,0,0])
        return colors


    def addMesh(self, mesh):
        self.merged_mesh += mesh # TODO: add up triangle meshes (does not work for t.geometry)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _id = self.scene.add_triangles(mesh)
        return _id

    def rayCast(self, T=np.eye(4)):
        R = T[0:3,0:3]
        origin = T[0:3,3]
        self.ray_img
        ray_img = np.einsum("ik,...k->...i", R, self.ray_img)
        rays = np.concatenate([np.array(origin)*self.origin_img_, ray_img], axis=-1)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        return ans, rays
    
    def computeDistance(self, query_points):
        query_points = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        distance = self.scene.compute_signed_distance(query_points)
        return distance





color_map = {
  0 : [0, 0, 0],
  1 : [245, 150, 100],
  2 : [245, 230, 100],
  3 : [150, 60, 30],
  4 : [180, 30, 80],
  5 : [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [125,125,125],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
  19: [250, 250, 250],
  20: [0, 250, 0]
}

# Create the custom color map
custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)

for i in range(256):
    if i in color_map:
        custom_colormap[i, 0, :] = color_map[i]
    else:
        # If the index is not defined in the color map, set it to black
        custom_colormap[i, 0, :] = [0, 0, 0]
custom_colormap = custom_colormap[...,::-1]

id_map = {
  0 : 0,     # "unlabeled"
  1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 20,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}


class Config:
    # Data specific params
    apply_pose = True
    min_range = 2.0
    max_range = 70.0


def cloud_to_image(points: np.ndarray, minx: float, maxx: float, miny: float, maxy: float, resolution: float) -> np.ndarray:
    """
    Converts a point cloud to an image.

    :param points: An array of points in the cloud, where each row represents a point.
                   The array shape can be (N, 3) or (N, 6).
                   If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                   If the shape is (N, 6), the last three columns represent the RGB color values for each point.
    :type points: ndarray
    :param minx: The minimum x-coordinate value of the cloud bounding box.
    :type minx: float
    :param maxx: The maximum x-coordinate value of the cloud bounding box.
    :type maxx: float
    :param miny: The minimum y-coordinate value of the cloud bounding box.
    :type miny: float
    :param maxy: The maximum y-coordinate value of the cloud bounding box.
    :type maxy: float
    :param resolution: The resolution of the image in units per pixel.
    :type resolution: float
    :return: An image array representing the point cloud, where each pixel contains the RGB color values
             of the corresponding point in the cloud.
    :rtype: ndarray
    :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
    """
    if points.shape[1] == 3:
        colors = np.array([255, 255, 255])
    else:
        colors = points[:, -3:]

    x = (points[:, 0] - minx) / resolution
    y = (maxy - points[:, 1]) / resolution
    pixel_x = np.floor(x).astype(np.uint)
    pixel_y = np.floor(y).astype(np.uint)

    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[pixel_y, pixel_x] = colors
    return image


class KITTIOdometryDataset:
    def __init__(self, kitti_root_dir: str, sequence: int, has_labels = True):
        """Simple KITTI DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """
        # Config stuff
        self.sequence = str(int(sequence)).zfill(2)
        self.config = Config()
        self.kitti_sequence_dir = os.path.join(kitti_root_dir, "sequences", self.sequence)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        # Read stuff
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))
        #self.poses = self.load_poses(os.path.join(kitti_root_dir, f"poses/{self.sequence}.txt"))
        self.poses = self.load_poses(os.path.join(self.kitti_sequence_dir, "poses.txt"))
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.has_labels = has_labels
        if has_labels:
            self.label_dir = os.path.join(self.kitti_sequence_dir, "labels/")
            self.label_files = sorted(glob.glob(self.label_dir + "*.label"))


    def __getitem__(self, idx):
        if self.has_labels:
            #print(self.scan_files[idx], self.scans(idx).shape, self.labels(idx).shape, self.label_files[idx])
            return *self.scans(idx), self.labels(idx), self.poses[idx]
            
        else:
            return *self.scans(idx), None, self.poses[idx]
            
    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):

        return self.read_point_cloud(idx, self.scan_files[idx], self.config)

    def labels(self, idx):
        file_name = self.label_files[idx]
        return self.read_label_(file_name)

    def read_point_cloud(self, idx: int, scan_file: str, config: Config):
        xyzi = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = xyzi[:, :-1]
        points = transform_points(points, self.poses[idx]) if config.apply_pose else points
        keys = xyzi[:, -1]
        return points, keys

    def read_label_(self, scan_file):
        labels = np.fromfile(scan_file, dtype=np.uint32)
        labels = labels.reshape((-1))
        #upper_half = labels >> 16      # get upper half for instances
        #lower_half = labels & 0xFFFF   # get lower half for semantics
        return labels

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right
        
        poses = pd.read_csv(poses_file, sep=" ", header=None).values
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)
                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict

from vdbfusion import VDBVolume



if __name__ == "__main__":
    ### Create Full Mesh
    kitti_root_dir="/home/hannes/data/semantic_THAB/"
    sequence = 1
    save_path = "/home/hannes/Desktop/THABBEV/kitti_sequence_{}".format(sequence)
    os.makedirs(save_path, exist_ok=True)
    dataset = KITTIOdometryDataset(kitti_root_dir=kitti_root_dir, sequence=sequence)
    # Create a VDB Volume to integrate scans
    vdb_volume = VDBVolume(voxel_size=0.1, sdf_trunc=0.5, space_carving=True)
    map_cloud = o3d.geometry.PointCloud()
    i = 0
    for scan, key, labels, pose in tqdm(dataset):
        labels = np.array([id_map[l] for l in labels])
        i += 1
        if i>50:
            break
        color = np.stack((key, labels/20, np.zeros_like(labels)),axis=-1)
        vdb_volume.integrate(scan, pose)

        cloud = o3d.geometry.PointCloud()

        cloud.points = o3d.utility.Vector3dVector(
            scan[labels>0])
        cloud.colors = o3d.utility.Vector3dVector(color[labels>0])

        map_cloud += cloud
    map_cloud.voxel_down_sample(voxel_size=0.05) 


    vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=True, min_weight=5.0)
    mesh = o3d.geometry.TriangleMesh()


    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_vertex_normals()

    pcd_tree = o3d.geometry.KDTreeFlann(map_cloud)
    query_points = np.asarray(mesh.vertices)
    colors = []
    for i in tqdm(range(query_points.shape[0])):
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(query_points[i:i+1,:].T, 1)
        colors.append(np.asarray(map_cloud.colors)[idx])
    colors = np.concatenate(colors)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(os.path.join(save_path,"kitti_vdbmesh_raw.ply"), mesh)
