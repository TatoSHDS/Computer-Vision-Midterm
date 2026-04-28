import numpy as np
import cv2
import json
import os
import matplotlib.image as mpimg
from estimate_projection import estimate_projection_matrix

def load_xyz(filepath):
    print(f"Loading {filepath}...")
    data = np.loadtxt(filepath)
    pts = data[:, :3]
    normals = data[:, 3:6]
    return pts, normals

def save_xyz(filepath, pts, colors):
    print(f"Saving {filepath}...")
    colors_alpha = np.hstack((colors, np.full((colors.shape[0], 1), 255))) # Add Alpha channel
    data = np.hstack((pts, colors_alpha))
    np.savetxt(filepath, data, fmt='%.6f %.6f %.6f %d %d %d %d')

def save_ply(filepath, pts, colors):
    print(f"Saving {filepath}...")
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
        f.write("end_header\n")
        for i in range(len(pts)):
            x, y, z = pts[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 255\n")

def get_camera_center(P):
    M = P[:, :3]
    p4 = P[:, 3]
    try:
        C = -np.linalg.inv(M) @ p4
        return C
    except np.linalg.LinAlgError:
        _, _, V = np.linalg.svd(P)
        C = V[-1]
        return C[:3] / C[3]

def build_z_buffer(depths, proj_x, proj_y, img_shape, window_size=5):
    h, w = img_shape[:2]
    valid = (proj_x >= 0) & (proj_x < w) & (proj_y >= 0) & (proj_y < h)
    u = proj_x[valid]
    v = proj_y[valid]
    d = depths[valid]
    
    z_buffer = np.full((h, w), 9999.0, dtype=np.float32)
    np.minimum.at(z_buffer, (v, u), d)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window_size, window_size))
    z_buffer_dilated = cv2.erode(z_buffer, kernel)
    return z_buffer_dilated

def main():
    if not os.path.exists("correspondences.json"):
        print("Error: correspondences.json not found! Please run pick_points.py first.")
        return

    with open("correspondences.json", "r") as f:
        data = json.load(f)

    pts, normals = load_xyz("7Images and xyz/Santa.xyz")
    num_pts = len(pts)
    accum_color = np.zeros((num_pts, 3))
    accum_weight = np.zeros(num_pts)
    
    images_dir = "7Images and xyz"
    
    for img_name, points in data.items():
        if len(points) < 6:
            print(f"Skipping {img_name}: Not enough points (need at least 6).")
            continue
            
        print(f"\nProcessing {img_name}...")
        img_path = os.path.join(images_dir, img_name)
        img = mpimg.imread(img_path)
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        img_shape = img.shape
        
        p2d = [p["2d"] for p in points]
        p3d = [p["3d"] for p in points]
        P = estimate_projection_matrix(p2d, p3d)
        
        C = get_camera_center(P)
        
        # View directions
        V_d = C - pts
        V_norm = np.linalg.norm(V_d, axis=1)
        V_d = V_d / (V_norm[:, np.newaxis] + 1e-8)
        
        # Dot product with normals
        dots = np.sum(normals * V_d, axis=1)
        
        # Front-face check
        front_facing = dots > 0
        
        # Projection
        pts_homo = np.hstack((pts, np.ones((num_pts, 1))))
        proj_homo = (P @ pts_homo.T).T
        depths = proj_homo[:, 2]
        
        valid_depth = depths > 0
        
        # Compute coordinates
        proj_x = np.zeros(num_pts, dtype=np.int32)
        proj_y = np.zeros(num_pts, dtype=np.int32)
        
        # Only compute for valid depth to avoid division by zero
        proj_x[valid_depth] = np.round(proj_homo[valid_depth, 0] / depths[valid_depth]).astype(np.int32)
        proj_y[valid_depth] = np.round(proj_homo[valid_depth, 1] / depths[valid_depth]).astype(np.int32)
        
        # Build Z-buffer using ONLY front-facing and valid depth points
        valid_for_z = front_facing & valid_depth
        z_buffer = build_z_buffer(depths[valid_for_z], proj_x[valid_for_z], proj_y[valid_for_z], img_shape, window_size=9)
        
        # Occlusion check and blending
        for i in range(num_pts):
            if not valid_for_z[i]:
                continue
            
            u, v = proj_x[i], proj_y[i]
            if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
                z_dist = depths[i]
                # Distance effect: allow a small tolerance for occlusion
                if z_dist <= z_buffer[v, u] + 0.05:
                    weight = dots[i] # View-dependent attenuation factor
                    color = img[v, u]
                    accum_color[i] += color[:3] * weight
                    accum_weight[i] += weight

    print("\nFinalizing colors...")
    final_colors = np.zeros((num_pts, 3), dtype=np.uint8)
    for i in range(num_pts):
        if accum_weight[i] > 0:
            final_colors[i] = np.clip(accum_color[i] / accum_weight[i], 0, 255).astype(np.uint8)
        else:
            final_colors[i] = [128, 128, 128] # Gray for unseen points
            
    save_xyz("Santa_Colorized.xyz", pts, final_colors)
    save_ply("Santa_Colorized.ply", pts, final_colors)
    print("Done! Check Santa_Colorized.ply in MeshLab.")

if __name__ == "__main__":
    main()
