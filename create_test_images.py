import numpy as np
from PIL import Image
import os

os.makedirs('d:/MyCode/MyCodeProject/PhthonProject/imageAlign/img_align_by_flow/test_images', exist_ok=True)

# Generate img1 (checkerboard or simple shapes)
img1 = np.zeros((400, 600, 3), dtype=np.uint8)
img1[100:300, 100:300] = [255, 0, 0] # Red square
img1[150:350, 350:550] = [0, 255, 0] # Green square

# Generate img2 (shifted img1)
img2 = np.zeros((400, 600, 3), dtype=np.uint8)
img2[120:320, 150:350] = [255, 0, 0] # Shifted (+20, +50)
img2[170:370, 400:600] = [0, 255, 0] # Shifted (+20, +50)

Image.fromarray(img1).save('d:/MyCode/MyCodeProject/PhthonProject/imageAlign/img_align_by_flow/test_images/img1.png')
Image.fromarray(img2).save('d:/MyCode/MyCodeProject/PhthonProject/imageAlign/img_align_by_flow/test_images/img2.png')
print('Test images generated.')
