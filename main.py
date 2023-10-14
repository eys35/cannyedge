import numpy as np
from PIL import Image, ImageDraw

# Test
print(np.random.randint(0, 100))

# 1) Transform into greyscale
def convert_to_grayscale(image_path, output_path):

    original_image = Image.open(image_path)
    rgb_array = np.array(original_image)
    grayscale_array = np.dot(rgb_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    output_image = Image.fromarray(grayscale_array)
    
    output_image.save(output_path)

def apply_gaussian_filter(image_path, output_path):
    image = Image.open(image_path).convert("L")
    grayscale_array = np.array(image)

    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

    kernel = kernel / np.sum(kernel)

    filtered_array = np.zeros_like(grayscale_array, dtype=np.float64)

    for i in range(1, grayscale_array.shape[0] - 1):
        for j in range(1, grayscale_array.shape[1] - 1):
            filtered_array[i, j] = np.sum(grayscale_array[i-1:i+2, j-1:j+2] * kernel)

    filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
    filtered_image.save(output_path)

def gradient_calculation(image_path):
    image = Image.open(image_path).convert("L")
    filtered_array = np.array(image)

    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]])
    rows, cols = filtered_array.shape
    Gx = np.zeros((rows, cols))
    Gy = np.zeros((rows, cols))
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            Gx[i, j] = np.sum(filtered_array[i-1:i+2, j-1:j+2] * Kx)
            Gy[i, j] = np.sum(filtered_array[i-1:i+2, j-1:j+2] * Ky)

    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255
    theta = np.arctan2(Gx, Gy)
    
    return G, theta


input_image_path = "emma.png" 
output_image_path = "greyscale.png"  # output path
gaussian_path = "gaussian output.png"  # output path
convert_to_grayscale(input_image_path, output_image_path)
apply_gaussian_filter(output_image_path, gaussian_path)
print(gradient_calculation(gaussian_path))