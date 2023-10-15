import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.image import imread
# input_image_path = "emma.png" 
input_image_path = input("Enter image file name: ")
output_image_path = "greyscale.png"  # output path
gaussian_path = "gaussian output.png"  # output path
gradient_path = "gradient.png" # output path
non_max_path = "non_max_supp.png" # output path
hysteresis_path= "hysteresis.png"

# Test
print(np.random.randint(0, 100))

# 1) Transform into greyscale
def convert_to_grayscale(image_path, output_path):

    original_image = Image.open(image_path)
    rgb_array = np.array(original_image)
    grayscale_array = np.dot(rgb_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    output_image = Image.fromarray(grayscale_array)
    
    output_image.save(output_path)
    return original_image

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

def gradient_calculation(image_path, output_path):
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
    
    gradient_image = Image.fromarray(G.astype(np.uint8))
    gradient_image.save(output_path)
    return G, theta

def non_max_suppression(gradient_magnitude, gradient_direction, output_path):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
 
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
 
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
 
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                efore_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
 
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
 
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    non_max_supp_image = Image.fromarray(output.astype(np.uint8))
    non_max_supp_image.save(output_path)
    return output

def double_thresholding(img,lowThresholdRatio=0.05, highThresholdRatio=0.09):
        M, N = img.shape

        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        res = np.zeros((M,N), dtype=np.int32)
        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
    
        return (res, weak, strong)

def hysteresis(output_path, img, weak, strong=255):
        M, N = img.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        hysteresis_img = Image.fromarray(img.astype(np.uint8))
        hysteresis_img.save(output_path)
        plt.imshow(hysteresis_img)
        return hysteresis_img


convert_to_grayscale(input_image_path, output_image_path)
apply_gaussian_filter(output_image_path, gaussian_path)
mag, dir = gradient_calculation(gaussian_path, gradient_path)
z = non_max_suppression(mag, dir, non_max_path)
res, weak, strong = double_thresholding(z)
hysteresis(hysteresis_path, res, weak, strong)

before = imread(input_image_path)
plt.imshow(before)
plt.axis('off')
plt.title("Before")
plt.show()
after = imread(hysteresis_path)
plt.imshow(after, cmap='gray')
plt.axis('off')
plt.title("After")
plt.show()