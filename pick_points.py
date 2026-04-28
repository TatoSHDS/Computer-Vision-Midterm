import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Features mapped to their exact 3D coordinates and visibility per image
FEATURES = {
    "Hat tip": {"3d": [0.217469, -13.860136, 68.332115], "images": [1, 2, 3, 4, 5, 6, 7]},
    "Nose tip": {"3d": [-0.226268, 10.566475, 39.553982], "images": [1, 2, 3, 4, 5]},
    "Left eye center": {"3d": [-3.689495, 6.203821, 40.159523], "images": [2, 4, 5]},
    "Right eye center": {"3d": [3.613763, 6.257575, 40.029438], "images": [1, 3, 5]},
    "Left mustache corner": {"3d": [-6.813026, 7.891991, 37.516628], "images": [2, 4, 5]},
    "Right mustache corner": {"3d": [6.324754, 8.109342, 37.526596], "images": [1, 3, 5]},
    "Sign top left": {"3d": [-10.866316, 13.730668, 20.599365], "images": [2, 3, 4, 5, 6]},
    "Sign top right": {"3d": [10.843328, 13.663549, 20.675194], "images": [1, 3, 4, 5, 7]},
    "Sign bottom left": {"3d": [-10.11687, 15.22287, 11.181825], "images": [2, 3, 4, 5, 6]},
    "Sign bottom right": {"3d": [10.153083, 15.153901, 11.216746], "images": [1, 3, 4, 5, 7]},
    "Left boot tip": {"3d": [-5.902934, 10.888931, 3.089641], "images": [2, 3, 4, 5]},
    "Right boot tip": {"3d": [5.807292, 10.872710, 3.081226], "images": [1, 3, 4, 5]},
    "Left hat-ear intersect.": {"3d": [-9.766281, -1.853524, 38.822239], "images": [2, 4, 5, 6]},
    "Right hat-ear intersect.": {"3d": [9.745612, -1.822872, 38.759541], "images": [1, 3, 5, 7]},
    "Back hair (middle)": {"3d": [-2.014375, -8.645645, 30.913752], "images": [1, 2, 6, 7]},
    "Left belt": {"3d": [-10.965766, -0.204803, 20.198553], "images": [2, 4, 6]},
    "Right belt": {"3d": [11.223559, 0.849981, 19.887234], "images": [1, 3, 7]},
    "Left boot back": {"3d": [-3.507453, -5.924526, 2.46912], "images": [2, 6, 7]},
    "Right boot back": {"3d": [3.497073, -5.947127, 2.436899], "images": [1, 6, 7]}
}

images_dir = "7Images and xyz"
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
image_files.sort()

correspondences = {}

for img_file in image_files:
    # Get image number (e.g., '01.jpg' -> 1)
    img_idx = int(img_file.split('.')[0])
    
    # Filter features that are visible in this specific image
    visible_features = []
    for f_name, f_data in FEATURES.items():
        if img_idx in f_data["images"]:
            visible_features.append((f_name, f_data["3d"]))
            
    if not visible_features:
        continue
        
    img_path = os.path.join(images_dir, img_file)
    img = mpimg.imread(img_path)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    
    # Try to maximize the window for better clicking accuracy
    try:
        fig.canvas.manager.window.state('zoomed')
    except Exception:
        pass
    
    print(f"\n--- Processing {img_file} ({len(visible_features)} features) ---")
    
    img_correspondences = []
    for f_name, f_3d in visible_features:
        # Update the title sequentially to tell the user what to click
        ax.set_title(f"Image {img_file} | Please click exactly on: {f_name}", fontsize=16, color="red", fontweight="bold")
        plt.draw()
        
        # Capture 1 point
        pt = plt.ginput(1, timeout=0, show_clicks=True)
        if pt:
            pt_2d = pt[0]
            print(f"Picked '{f_name}' at ({pt_2d[0]:.2f}, {pt_2d[1]:.2f})")
            img_correspondences.append({
                "name": f_name,
                "2d": [pt_2d[0], pt_2d[1]],
                "3d": f_3d
            })
        else:
            print(f"Skipped '{f_name}'")
            
    plt.close(fig)
    correspondences[img_file] = img_correspondences

with open("correspondences.json", "w") as f:
    json.dump(correspondences, f, indent=4)

print("\nSaved correspondences to correspondences.json.")

