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

input_image_path = "emma.png" 
output_image_path = "output.png"  # output path
convert_to_grayscale(input_image_path, output_image_path)