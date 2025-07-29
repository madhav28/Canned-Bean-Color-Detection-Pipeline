from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage import io, color, img_as_ubyte
from skimage.io import imsave
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import joblib
import cv2
import os
import warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image*")


def extract_and_calibrate_lab():
    # extraction start
    valid_exts = ('.jpg', '.jpeg', '.png')
    img_path = [f"./input/images/{file}" for file in os.listdir("./input/images") if file.lower().endswith(valid_exts)]
    seg_path = [path.replace("/input/images", "/output/segmentation") for path in img_path]
    filenames = [path.split('/')[-1].split('.')[0] for path in img_path]

    lab_values = {}
    for i in tqdm(range(len(filenames)), desc="⏳ Extracting LAB"):
        filename = filenames[i]
        p1 = img_path[i]
        p2 = seg_path[i]
        rgb_image = io.imread(p1)
        mask_image = io.imread(p2)
        rgb_image = np.rot90(rgb_image, k=1)
        
        if rgb_image.shape != mask_image.shape:
            print("❌ Error: Shape mismatch between rgb_image and mask_image. Please verify the images and rerun the pipeline.")

        if mask_image.dtype == np.float64:
            mask_image = (mask_image * 255).astype(np.uint8)

        lower = np.array([0, 245, 0])
        upper = np.array([10, 255, 10])
        green_mask = (
            (mask_image[:, :, 0] >= lower[0]) & (mask_image[:, :, 0] <= upper[0]) &
            (mask_image[:, :, 1] >= lower[1]) & (mask_image[:, :, 1] <= upper[1]) &
            (mask_image[:, :, 2] >= lower[2]) & (mask_image[:, :, 2] <= upper[2])
        )
        green_mask_visual = green_mask.astype(np.uint8) * 255

        if not np.any(green_mask):
            print(f"❌ No segmentation mask found in {p2}, skipping.")
            continue

        lab_image = color.rgb2lab(rgb_image)

        L_masked = lab_image[:, :, 0][green_mask]
        A_masked = lab_image[:, :, 1][green_mask]
        B_masked = lab_image[:, :, 2][green_mask]

        lab_masked = np.stack([L_masked, A_masked, B_masked], axis=1)

        lab_values[filename] = lab_masked
    # extraction end

    # calibration start
    checker_path = r"./input/color_checker.JPG"
    patches_dir = r"./output/patches"
    patch_size = (100, 100)
    rows = 4
    cols = 6
    os.makedirs(patches_dir, exist_ok=True)

    img = cv2.imread(checker_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    cell_h = img_h // rows
    cell_w = img_w // cols
    crop_h, crop_w = patch_size

    patch_id = 1
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h + (cell_h - crop_h) // 2
            x0 = c * cell_w + (cell_w - crop_w) // 2
            patch = img_rgb[y0:y0 + crop_h, x0:x0 + crop_w]

            patch_name = f"patch_{patch_id:02d}.png"
            imsave(os.path.join(patches_dir, patch_name), img_as_ubyte(patch))
            patch_id += 1

    print(f"🧩 Extracted and Saved {patch_id - 1} Patches to: {patches_dir}")


    lab_patch_path = r'./output/patches/lab_patch.csv'

    lab_summary = []

    for filename in os.listdir(patches_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image_path = os.path.join(patches_dir, filename)
                image = io.imread(image_path)

                lab_image = color.rgb2lab(image)

                L_values = lab_image[:, :, 0].flatten()
                a_values = lab_image[:, :, 1].flatten()
                b_values = lab_image[:, :, 2].flatten()

                stats = {
                    'Image': filename,
                    'L': np.mean(L_values),
                    'A': np.mean(a_values),
                    'B': np.mean(b_values),
                }

                lab_summary.append(stats)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df_summary = pd.DataFrame(lab_summary)
    df_summary = df_summary.sort_values(by='Image')
    df_summary.to_csv(lab_patch_path, index=False)


    y = pd.read_csv(r"./assets/ref_checker_LAB.csv")
    y = y[['L', 'A', 'B']]
    X =  pd.read_csv(r"./output/patches/lab_patch.csv")
    X = X[['L', 'A', 'B']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_components = 3
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_train, y_train)

    names = []
    L_initial = []
    A_initial = []
    B_initial = []
    L_calib = []
    A_calib = []
    B_calib = []
    for key in tqdm(lab_values, desc="⏳ Calibrating LAB"):
        names.append(key)
        lab = lab_values[key]
        L_initial.append(np.mean(lab[:, 0]))
        A_initial.append(np.mean(lab[:, 1]))
        B_initial.append(np.mean(lab[:, 2]))
        lab = pd.DataFrame(lab, columns=X_train.columns)
        lab_calib = pls_model.predict(lab)
        L_calib.append(np.mean(lab_calib[:, 0]))
        A_calib.append(np.mean(lab_calib[:, 1]))
        B_calib.append(np.mean(lab_calib[:, 2]))

    df = {'Name': names, 'L_initial': L_initial, 'A_initial': A_initial, 'B_initial': B_initial,
          'L_calib': L_calib, 'A_calib': A_calib, 'B_calib': B_calib}
    df = pd.DataFrame(df)
    df.to_csv('./output/results.csv', index=False)

    # y_pred = pls_model.predict(X_test)
    # r_squared = pls_model.score(X_test, y_test)
    # print(f"R-Squared: {r_squared}")
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
    
    # calibration end