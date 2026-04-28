import numpy as np

def normalize_2d(pts):
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    dist = np.sqrt(np.sum(shifted**2, axis=1))
    scale = np.sqrt(2) / (np.mean(dist) + 1e-8)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    return T, (shifted * scale)

def normalize_3d(pts):
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    dist = np.sqrt(np.sum(shifted**2, axis=1))
    scale = np.sqrt(3) / (np.mean(dist) + 1e-8)
    U = np.array([
        [scale, 0, 0, -scale * centroid[0]],
        [0, scale, 0, -scale * centroid[1]],
        [0, 0, scale, -scale * centroid[2]],
        [0, 0, 0, 1]
    ])
    return U, (shifted * scale)

def dlt(pts_2d, pts_3d):
    n = pts_2d.shape[0]
    A = []
    for i in range(n):
        X, Y, Z = pts_3d[i]
        u, v = pts_2d[i]
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 4)
    return P

def estimate_projection_matrix(pts_2d, pts_3d):
    pts_2d = np.array(pts_2d)
    pts_3d = np.array(pts_3d)
    T, norm_2d = normalize_2d(pts_2d)
    U, norm_3d = normalize_3d(pts_3d)
    
    P_norm = dlt(norm_2d, norm_3d)
    
    P = np.linalg.inv(T) @ P_norm @ U
    P = P / P[-1, -1]
    return P

def compute_reprojection_error(P, pts_2d, pts_3d):
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    proj = P @ pts_3d_homo.T
    proj = proj[:2, :] / proj[2, :]
    proj = proj.T
    errors = np.linalg.norm(proj - pts_2d, axis=1)
    return np.mean(errors)

if __name__ == "__main__":
    import json
    import os
    if os.path.exists("correspondences.json"):
        with open("correspondences.json", "r") as f:
            data = json.load(f)
        for img_name, points in data.items():
            if len(points) < 6:
                print(f"{img_name} has less than 6 points. Skipping evaluation.")
                continue
            p2d = [p["2d"] for p in points]
            p3d = [p["3d"] for p in points]
            P = estimate_projection_matrix(p2d, p3d)
            err = compute_reprojection_error(P, p2d, p3d)
            print(f"{img_name}: Average Reprojection Error = {err:.4f} pixels")
