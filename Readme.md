# 3D Model Colorization via Multi-View Image Re-projection

**NTUST — Computer Vision and Applications (CI5336701, 2026 Spring)**

A full multi-view 3D colorization pipeline that paints realistic, view-consistent color onto a raw 3D point cloud using 7 uncalibrated photographs taken from different angles with different cameras.

---

## Problem Statement

Given:
- A raw, colorless 3D point cloud of a garden gnome figurine (`Santa.xyz`, ~95,405 vertices), where each row stores `X Y Z Nx Ny Nz` (position + surface normal).
- 7 photographs taken by two different cameras (iPhone and Sony DSLR), each with a different intrinsic matrix **K** and extrinsic pose **[R|t]**.

The task is to:
1. Estimate the **projection matrix P = K[R|t]** for each image from manually selected 2D↔3D correspondences.
2. Project all 3D vertices into each image, handling visibility, back-faces, and occlusion correctly.
3. Blend color samples across multiple views into a single consistent RGBA point cloud.

No prior camera calibration is provided or required — everything is recovered from point correspondences.

---

## Pipeline Overview

```
Santa.xyz (geometry + normals)
        │
        ▼
[pick_points.py] ──── manually click 2D↔3D landmarks ──→ correspondences.json
        │
        ▼
[estimate_projection.py] ──── Normalized DLT per image ──→ P₁ … P₇
        │
        ▼
[colorize_model.py]
  ├── for each image:
  │    ├── extract camera center C from P
  │    ├── front-face check  (normal · view_dir > 0)
  │    ├── depth validity    (projected depth > 0)
  │    ├── Z-buffer occlusion (dilated depth buffer)
  │    └── accumulate color  (weighted by normal·view_dir)
  └── normalize accumulated color → Santa_Colorized.ply / .xyz
```

---

## Step 1 — Point Correspondence Collection (`pick_points.py`)

### Landmark design

19 anatomically distinct landmarks were chosen from the figurine and their exact 3D coordinates were measured in MeshLab. Selection criteria:

- **Spatial spread**: features span Z ≈ 3 (boot tips) to Z ≈ 68 (hat tip), and X from −11 to +11, ensuring the point set is non-planar and non-collinear — a necessary condition for a numerically well-conditioned DLT.
- **Unambiguous localization**: corners (sign corners, mustache corners, boot tips) and intersections (hat-ear junction) that can be clicked with sub-pixel precision.
- **Symmetric pairs**: left/right pairs on symmetric features (eyes, belt, boots, sign corners) give bilateral coverage for front, side, and rear views.
- **Visibility metadata**: each landmark stores which images it is visible in. Only landmarks visible in a given image are collected for that image — avoiding impossible or misleading picks.

| Feature | 3D (X, Y, Z) | Visible in images |
|---|---|---|
| Hat tip | (0.217, −13.860, 68.332) | 1–7 (all) |
| Nose tip | (−0.226, 10.566, 39.554) | 1,2,3,4,5 |
| Left / Right eye center | (±3.6, 6.2, 40.1) | side views |
| Left / Right mustache corner | (±6.5, 8.0, 37.5) | front/side views |
| Sign top-left / top-right | (±10.8, 13.7, 20.6) | front/side views |
| Sign bottom-left / bottom-right | (±10.1, 15.2, 11.2) | front/side views |
| Left / Right boot tip | (±5.9, 10.9, 3.1) | front/side views |
| Left / Right hat-ear intersect. | (±9.8, −1.8, 38.8) | side/rear views |
| Back hair (middle) | (−2.0, −8.6, 30.9) | rear views |
| Left / Right belt | (±11.1, 0.3, 20.0) | side views |
| Left / Right boot back | (±3.5, −5.9, 2.5) | rear views |

### Interactive picking

For each image the script filters to only its visible landmarks, displays the full-resolution image in a maximized Matplotlib window, and sequentially prompts the user with the landmark name in the title bar. One click per landmark is captured via `plt.ginput(1, timeout=0, show_clicks=True)`. The resulting `correspondences.json` stores `{name, 2d: [u, v], 3d: [X, Y, Z]}` per landmark per image.

---

## Step 2 — Projection Matrix Estimation (`estimate_projection.py`)

### The DLT formulation

A 3×4 projection matrix **P** maps homogeneous world points to image coordinates:

```
λ · [u, v, 1]ᵀ = P · [X, Y, Z, 1]ᵀ
```

Cross-multiplying to eliminate the unknown scale λ gives two linear equations per correspondence in the 12 entries of P:

```
[-X, -Y, -Z, -1,  0,  0,  0,  0,  u·X,  u·Y,  u·Z,  u] · p = 0
[ 0,  0,  0,  0, -X, -Y, -Z, -1,  v·X,  v·Y,  v·Z,  v] · p = 0
```

Stacking n pairs forms a (2n × 12) matrix **A**. The solution **p** is the right singular vector of A with the smallest singular value:

```python
_, _, V = np.linalg.svd(A)
P = V[-1].reshape(3, 4)
```

A minimum of 6 correspondences is required (11 DOF, 2 equations each).

### Hartley normalization

Raw pixel coordinates (~0–3000) and metric 3D coordinates (~−11 to 68) differ by orders of magnitude. Directly filling A with these values creates wildly unbalanced rows, causing the SVD to find a numerically poor solution. Hartley normalization is applied first:

**2D (`normalize_2d`)** — shift points to their centroid, then scale uniformly so the *average* distance from the origin becomes √2:
```python
scale = np.sqrt(2) / np.mean(dist)
T = [[scale, 0, -scale*cx],
     [0, scale, -scale*cy],
     [0,     0,          1]]
```

**3D (`normalize_3d`)** — same idea in 3D, target average distance √3:
```python
scale = np.sqrt(3) / np.mean(dist)
U = [[scale, 0, 0, -scale*cX],
     [0, scale, 0, -scale*cY],
     [0, 0, scale, -scale*cZ],
     [0,     0, 0,          1]]
```

After solving P_norm on the normalized point sets, the physical-space matrix is recovered by undoing the transforms:
```python
P = np.linalg.inv(T) @ P_norm @ U
P = P / P[-1, -1]   # canonical scale
```

### Reprojection error

Quality is verified by re-projecting each training point through P and measuring the mean pixel distance to the picked 2D position:
```python
proj = P @ pts_3d_homo.T
proj = (proj[:2] / proj[2]).T
error = np.mean(np.linalg.norm(proj - pts_2d, axis=1))
```
Low error (< ~3 px) confirms both accurate picking and a well-conditioned solve.

---

## Step 3 — Colorization (`colorize_model.py`)

For every image, each of the 95,405 vertices is passed through a four-stage visibility pipeline. Two global buffers `accum_color` (N×3) and `accum_weight` (N) are accumulated across all images.

### Stage 1 — Camera center extraction

The camera center **C** is the null-space of P. For non-degenerate P with left 3×3 block M:
```python
C = -np.linalg.inv(M) @ P[:, 3]
```
Fallback for degenerate cases: right singular vector of P divided by its 4th component.

### Stage 2 — Front-face check

The unit view direction from each vertex to the camera is computed, then dotted with the stored surface normal:
```python
V_d = (C - pts) / ||C - pts||
dots = np.sum(normals * V_d, axis=1)
front_facing = dots > 0
```
`dots[i] > 0` means the surface normal points toward the camera — the vertex faces the camera. Vertices with `dots[i] ≤ 0` are back-faces and are immediately excluded. This exploits the per-vertex normals already stored in the `.xyz` file.

### Stage 3 — Depth validity and projection

All vertices are projected via vectorized homogeneous multiplication:
```python
proj_homo = (P @ pts_homo.T).T      # (N×3)
depths = proj_homo[:, 2]
```
Only vertices with `depths > 0` (in front of the camera) proceed. Pixel coordinates are:
```python
proj_x = round(proj_homo[:, 0] / depths)
proj_y = round(proj_homo[:, 1] / depths)
```

### Stage 4 — Z-buffer occlusion with morphological dilation

A depth buffer is built from all front-facing, valid-depth vertices using a per-pixel minimum:
```python
z_buffer = np.full((H, W), 9999.0)
np.minimum.at(z_buffer, (proj_y, proj_x), depths)
```

Because the point cloud is sparse, many pixels have no vertex, leaving the z-buffer full of 9999 gaps. A vertex behind a nearby surface could incorrectly pass the occlusion test through such a gap. This is fixed by applying **morphological erosion** with a 9×9 elliptical kernel — equivalent to dilating the minimum depth outward to fill gaps:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
z_buffer = cv2.erode(z_buffer, kernel)
```

A vertex passes occlusion if:
```python
depths[i] <= z_buffer[v, u] + 0.05
```
The +0.05 tolerance prevents self-occlusion from floating-point rounding at the same depth layer.

### Color accumulation with view-dependent weighting

Every vertex that passes all four stages samples the image color at its projected pixel and accumulates it with weight = `dots[i]` (the normal–view cosine):
```python
accum_color[i]  += img[v, u, :3] * dots[i]
accum_weight[i] += dots[i]
```

This is **view-dependent weighted blending**: views where the camera faces the surface nearly head-on (`dots ≈ 1`) contribute fully, while grazing-angle views (`dots ≈ 0`) contribute almost nothing. This is analogous to cosine-weighted irradiance averaging and produces smooth, artifact-reduced results at view-seam boundaries.

After all images are processed:
```python
final_colors[i] = clip(accum_color[i] / accum_weight[i], 0, 255)
```
Vertices with zero accumulated weight (never visible from any image) get neutral gray `(128, 128, 128)`.

---

## Requirements

```
numpy
opencv-python
matplotlib
```

---

## Usage

```bash
# Step 1: Collect 2D-3D correspondences (requires a display)
python pick_points.py

# Step 2 (optional): Verify reprojection errors
python estimate_projection.py

# Step 3: Run colorization → Santa_Colorized.ply
python colorize_model.py
```

Open `Santa_Colorized.ply` in **MeshLab** to inspect the result. Or you can also copy-paste the result from `Santa_Colorized.xyz` into `SantaTriangle4Test.ply` in the part that says "paste your result here", make sure to leave no empty lines. A sample output that has already been done can be seen into `SantaTriangle4Test_Output.ply`, you can also check that out on **MeshLab**. 

---

## Key Techniques Summary

| Technique | Why it matters |
|---|---|
| Normalized DLT (Hartley) | Balances scale disparities between pixel and metric coordinates for a numerically stable SVD solve |
| Reprojection error check | Quantitatively validates each P before colorization; flags bad picks |
| Surface normal front-face rejection | Prevents rear/interior vertices from being incorrectly colored with background pixels |
| Depth validity check | Rejects vertices behind the camera plane (negative depth) |
| Dilated Z-buffer occlusion | Handles sparsity gaps in the point cloud by morphologically propagating minimum depths |
| View-dependent cosine weighting | Prioritizes head-on views in the color blend, suppressing grazing-angle artifacts |
| Neutral-gray fallback | Keeps unseen vertices visually distinct rather than assigning arbitrary color |