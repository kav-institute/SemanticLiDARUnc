import os
import numpy as np
from tqdm import tqdm
from contextlib import closing
from ouster import client
from ouster import osf
import argparse


def convert(osf_path, config, save_path, col=1023):
    txt_save_path = os.path.join(save_path, "poses.txt")
    point_save_path = os.path.join(save_path, "velodyne")
    os.makedirs(point_save_path, exist_ok=True)

    with open(config, 'r') as f:
        metadata = client.SensorInfo(f.read())
        #source = osf.Scans(osf_path)#pcap.Pcap(pcap_path, metadata)
        load_scan = lambda:  osf.Scans(osf_path)
        j = 0

    all_poses = []
    with closing(load_scan()) as stream:
        all_poses = []

        for i, scan in tqdm(enumerate(stream)):
            if i<=1:
                continue
            
            xyzlut = client.XYZLut(metadata)
            xyz = xyzlut(scan)
            xyz = client.destagger(stream.metadata, xyz)
            h, w, c = xyz.shape
            reflectivity_field = scan.field(client.ChanField.REFLECTIVITY)
            reflectivity_img = client.destagger(
                stream.metadata, reflectivity_field)

            ts = scan.timestamp[col]
            name = str(np.uint64(ts))

            T = scan.pose[col,...]
            
            pc = np.concatenate([xyz, reflectivity_img[...,np.newaxis]], axis=-1).reshape((-1,4)).astype(np.float32)
            pc.tofile(os.path.join(point_save_path, name + ".bin"))
            #print(os.path.join(point_save_path, name + ".bin"))
            pose = [T[0,0], T[0,1], T[0,2], T[0,3],
                        T[1,0], T[1,1], T[1,2], T[1,3],
                        T[2,0], T[2,1], T[2,2], T[2,3]]

            all_poses.append(pose)
    np.savetxt(txt_save_path, np.array(all_poses))

def main(args):
    convert(args.osf_path, args.config_path, args.save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--save_path', type=str,
                        help='Path to save the Ouster Scan in KITTI format.')
    parser.add_argument('--osf_path', type=str,
                        help='Path to Ouster Scan in OSF format.')
    parser.add_argument('--config_path', type=str,
                        help='Path to Ouster Config in json format.')
    args = parser.parse_args()

    main(args)
