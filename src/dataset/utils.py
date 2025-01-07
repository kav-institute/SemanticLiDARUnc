import numpy as np
import cv2

def rotate_z(point_cloud, angle):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply the rotation matrix to the point cloud
    rotated_cloud = np.dot(point_cloud, rotation_matrix)
    
    return rotated_cloud


def rotate_equirectangular_image(image, angle):
    # Calculate the shift amount based on the angle of rotation
    shift_amount = int(round((angle / (2 * np.pi)) * image.shape[1]))

    # Roll the image around the width axis
    rotated_image = np.roll(image, shift_amount, axis=1)

    return rotated_image

def build_normal_xyz(xyz, norm_factor=0.25, ksize = 3):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
    '''
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    #build cross product
    normal = -np.dstack((Syx*Szy - Szx*Syy,
                        Szx*Sxy - Szy*Sxx,
                        Sxx*Syy - Syx*Sxy))

    # normalize corss product
    n = np.linalg.norm(normal, axis=2)+1e-10
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    return normal

def to_deflection_coordinates(x,y,z):
    # To cylindrical
    p = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # To spherical   
    theta = -np.arctan2(p, z) + np.pi/2
    return phi, theta

def spherical_projection(pc, height=64, width=2048, theta_range=None, th=1.0, sort_largest_first=False, bins_h=None, max_range=None):
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*C
    Returns:
        pj_img: projected spherical iamges, shape: h*w*C
    '''

    # filter all small range values to avoid overflows in theta min max calculation
    #if isinstance(theta_range, type(None)):
        
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    arr1inds = r.argsort()
    if sort_largest_first:
        pc = pc[arr1inds]
    else:
        pc = pc[arr1inds[::-1]]
    #pc = pc[arr1inds]
    # r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    # if not isinstance(max_range,type(None)):
    #     indices = np.where((r > th)*(r<=max_range))
    # else:
    #     indices = np.where(r > th)
    # pc = pc[indices]
        
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
    phi, theta = to_deflection_coordinates(x,y,z)

    #indices = np.where(r > th)
    if isinstance(theta_range, type(None)):
        theta_min, theta_max = [theta.min(), theta.max()]
    else: 
        theta_min, theta_max = theta_range
        
    phi_min, phi_max = [-np.pi, np.pi]
    
    # assuming uniform distribution of rays
    if isinstance(bins_h, type(None)):
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
        
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]
    
    theta_img = np.stack(width*[bins_h], axis=-1)
    phi_img = np.stack(height*[bins_w], axis=0)

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1
    
    pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)

    
    pj_img[idx_h, idx_w, :] = pc

   
    alpha = np.sqrt(np.square(theta_img)+np.square(phi_img))
   
    return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max) 
