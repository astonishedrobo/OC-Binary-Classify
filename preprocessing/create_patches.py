import cv2
import numpy as np
import os
from tqdm import tqdm

def generate_patches(image_path, patch_size=(960, 960), stride=(480, 480)):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Calculate the number of patches in each dimension
    num_patches_x = (width - patch_size[1]) // stride[1] + 1
    num_patches_y = (height - patch_size[0]) // stride[0] + 1

    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate patch coordinates
            start_x = j * stride[1]
            start_y = i * stride[0]
            end_x = start_x + patch_size[1]
            end_y = start_y + patch_size[0]

            # Check if the patch goes out of image dimensions
            if end_x > width:
                start_x = width - patch_size[1]
                end_x = width
            if end_y > height:
                start_y = height - patch_size[0]
                end_y = height

            # Extract the patch from the image
            patch = image[start_y:end_y, start_x:end_x]
            patch = cv2.resize(patch, (512, 512))
            patches.append(patch)

    return patches

def process_patients(directory, save_dir, patch_size=(960, 960), stride=(480, 480)):
    for pid_dir in tqdm(os.listdir(directory)):
        pid_path = os.path.join(directory, pid_dir)
        save_path = os.path.join(save_dir, pid_dir)
        
        if os.path.isdir(pid_path):
            os.makedirs(save_path, exist_ok=True)
            
            for image_file in os.listdir(pid_path):
                image_path = os.path.join(pid_path, image_file)
                
                if os.path.isfile(image_path):
                    patches = generate_patches(image_path, patch_size, stride)
                    
                    # Save patches in the same patient directory
                    for i, patch in enumerate(patches):
                        patch_name = f"{os.path.splitext(image_file)[0]}_{i}.png"
                        # print(os.path.join(save_path, patch_name))
                        cv2.imwrite(os.path.join(save_path, patch_name), patch)

def process_directory(directory, save_dir, patch_size=(960, 960), stride=(480, 480)):
    for oc_type in os.listdir(directory):
        if oc_type == "OSCC":
            for subtype in ["WD", "PD", "MD"]:
                oc_type_path = os.path.join(directory, oc_type, subtype)
                save_dir_ = os.path.join(save_dir, oc_type, subtype)
                print("Processing:", oc_type, subtype)

                if os.path.isdir(oc_type_path):
                    os.makedirs(save_dir, exist_ok=True)
                    process_patients(oc_type_path, save_dir_, patch_size, stride)
        else:
            oc_type_path = os.path.join(directory, oc_type)
            save_dir_ = os.path.join(save_dir, oc_type)
            print("Processing:", oc_type)
        
            if os.path.isdir(oc_type_path):
                os.makedirs(save_dir, exist_ok=True)
                process_patients(oc_type_path, save_dir_, patch_size, stride)

directory = "/media/KutumLabGPU/split_data_png_new/train"
save_dir = "/media/KutumLabGPU/oc-bin-cls-new/train"
patch_size = (960, 960)
stride = (480, 480)
process_directory(directory, save_dir, patch_size, stride)