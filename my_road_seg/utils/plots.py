import os
import cv2
import torch
import numpy as np 
def plot_image_file(img_file):
    img = cv2.imread(img_file)
    #output_file = os.path.join('outputs', os.path.basename(img_file))
    return img

def make_color_label_file(label_file):
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)  # (H, W) shape
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3) shape
    output_file = os.path.join('outputs', os.path.basename(label_file).replace('.png', '_label.png'))
    colors = [  # [B, G, R] value
        [0, 0, 0],         # 0: background
        [144, 124, 226],   # 1: motorway
        [172, 192, 251],   # 2: trunk
        [161, 215, 253],   # 3: primary
        [187, 250, 246],   # 4: secondary
        [255, 255, 255],   # 5: tertiary
        [49, 238, 75],     # 6: path
        [173, 173, 173],   # 7: under construction
        [255, 85, 170],    # 8: train guideway
        [234, 232, 120]    # 9: airplane runway
    ]
    for i in range(10):
        color_label[label == i] = colors[i]
    cv2.imwrite(output_file, color_label)
    output_file = os.path.join('outputs', os.path.basename(label_file).replace('.png', '_label.png'))
    return color_label


def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    if torch. is_tensor(img):
        img = img.mul(255.0).cpu().numpy().transepos(1,2,0).astype(np.uint8)
        
    if label is not None:
        if torch.is_tensor(label):
            label=label.cpu().numpy().astype(np.uint8)
        color_label = make_color_label_file(label_file)
        label = color_label 
    img =  cv2.addWeighted(img, 1, color_label, alpha, 0)
    cv2.imwrite(save_file, img)
    
    
    

if __name__ == "__main__":
    img = 'data/kari-road/train/images/BLD11166_PS3_K3A_NIA0390.png'
    label_file = '/home/ubuntu/my_road_seg/data/kari-road/train/labels/BLD11166_PS3_K3A_NIA0390.png'
    a=plot_image_file(img)
    b=plot_image_file(label_file)

    
    
    plot_image(b,a)
    