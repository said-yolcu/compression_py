import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy import optimize
import numpy as np
import math
import argparse
from PIL import Image
import sys
import os

# Manipulate channels


def get_greyscale_image(img):
    return np.mean(img[:, :, :2], 2)


def extract_rgb(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def assemble_rbg(img_r, img_g, img_b):
    shape = (img_r.shape[0], img_r.shape[1], 1)
    return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape),
                           np.reshape(img_b, shape)), axis=2)

# Transformations

# Return an image that has the size as factor times input img's size


def reduce(img, factor):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(
                img[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return result

# Rotates the image in positive direction (counter clockwise)


def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

# Return the input img if direction is 1, else flip the image vertically
# The flipping axis is the x axis


def flip(img, direction):
    return img[::direction, :]


def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast*rotate(flip(img, direction), angle) + brightness

# Contrast and brightness


def find_contrast_and_brightness1(D, S):
    # Fix the contrast and only fit the brightness
    contrast = 0.75
    brightness = (np.sum(D - contrast*S)) / D.size
    return contrast, brightness


def find_contrast_and_brightness2(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate(
        (np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]

# Compression for greyscale images


def generate_all_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = reduce(img[k*step:k*step+source_size, l *
                       step:l*step+source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append(
                    (k, l, direction, angle, apply_transformation(S, direction, angle)))
    return transformed_blocks


def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_all_transformed_blocks(
        img, source_size, destination_size, step)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i*destination_size:(i+1)*destination_size,
                    j*destination_size:(j+1)*destination_size]
            # Test all possible transformations and take the best one
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness2(D, S)
                S = contrast*S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (
                        k, l, direction, angle, contrast, brightness)
    return transformations


def decompress(transformations, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                S = reduce(
                    iterations[-1][k*step:k*step+source_size, l*step:l*step+source_size], factor)
                D = apply_transformation(S, flip, angle, contrast, brightness)
                cur_img[i*destination_size:(i+1)*destination_size,
                        j*destination_size:(j+1)*destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations

# Compression for color images


def reduce_rgb(img, factor):
    img_r, img_g, img_b = extract_rgb(img)
    img_r = reduce(img_r, factor)
    img_g = reduce(img_g, factor)
    img_b = reduce(img_b, factor)
    return assemble_rbg(img_r, img_g, img_b)


def compress_rgb(img, source_size, destination_size, step):
    img_r, img_g, img_b = extract_rgb(img)
    return [compress(img_r, source_size, destination_size, step),
            compress(img_g, source_size, destination_size, step),
            compress(img_b, source_size, destination_size, step)]


def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
    img_r = decompress(
        transformations[0], source_size, destination_size, step, nb_iter)[-1]
    img_g = decompress(
        transformations[1], source_size, destination_size, step, nb_iter)[-1]
    img_b = decompress(
        transformations[2], source_size, destination_size, step, nb_iter)[-1]
    return assemble_rbg(img_r, img_g, img_b)

# Plot


def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(
                str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

# Parameters


directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle]
              for direction in directions for angle in angles]

# Tests


def test_greyscale(img_name, typ):
    img = []
    if typ == "png":
        img = np.asarray(Image.open(img_name))
    elif typ == "gif":
        img = mpimg.imread(img_name)

    img = get_greyscale_image(img)
    img = reduce(img, 4)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    transformations = compress(img, 8, 4, 8)
    iterations = decompress(transformations, 8, 4, 8)
    plot_iterations(iterations, img)
    plt.show()


def test_rgb(img_name, typ):
    img = []
    if typ == "png":
        img = np.asarray(Image.open(img_name))
    elif typ == "gif":
        img = mpimg.imread(img_name)

    img = reduce_rgb(img, 8)
    transformations = compress_rgb(img, 8, 4, 8)
    retrieved_img = decompress_rgb(transformations, 8, 4, 8)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
    plt.show()


def test():
    # Images
    img = mpimg.imread("monkey.gif")
    img1 = flip(img, 1)
    img2 = flip(img, -1)
    img3 = rotate(img, 90)
    img4 = rotate(img, 180)
    img5 = rotate(img, 270)

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 3)

    # Display the first image
    axs[0][0].imshow(img)
    axs[0][0].set_title("original image")

    # Flipped images
    axs[0][1].imshow(img1)
    axs[0][1].set_title("flipped with 1")

    axs[0][2].imshow(img2)
    axs[0][2].set_title("flipped with -1")

    # Rotated images
    axs[1][0].imshow(img3)
    axs[1][0].set_title("rotated 90")

    axs[1][1].imshow(img4)
    axs[1][1].set_title("rotated 180")

    axs[1][2].imshow(img5)
    axs[1][2].set_title("rotated 270")

    plt.show()


def get_size(filename):
    stat = os.stat(filename)
    size = stat.st_size

    return size


def rl_encode(img, redClar=1):

    
    less_img = [[0 for j in range(len(img[0]))] for i in range(len(img))]

    for i in range(len(img)):
        for j in range(len(img[0])):
            less_img[i][j] = img[i][j]//redClar

    less_img = np.array(less_img)

    

    # Flatten, then compress the image
    encoded = []
    count = 0
    sum = 0
    prev = None
    fimg = less_img.flatten()
    different_colors = False

    for pixel in fimg:
        if prev == None:
            prev = pixel
            count += 1
            sum = pixel
        elif different_colors:
            if count < 3:
                prev = pixel
                sum += pixel
                count += 1
            else:  # count >= 3
                prev = pixel
                avg = sum // count
                # print("line 330 encoding", (count, avg))
                encoded.append((count, avg))
                sum = pixel
                count = 1
                different_colors = False
        elif prev != pixel:

            if count < 3:
                different_colors = True
                prev = pixel
                sum += pixel
                count += 1
            else:
                different_colors = False
                # print("line 344 encoding", (count, prev))
                encoded.append((count, prev))
                prev = pixel
                count = 1
                sum = pixel
        else:  # prev == pixel
            count += 1
            sum += pixel

    if different_colors:
        avg = sum//count
        # print("line 355 encoding", (count, avg))
        encoded.append((count, avg))
    else:
        # print("line 358 encoding", (count, prev))
        encoded.append((count, prev))

    # print("length of fimg is ", len(fimg))

    x_sum = 0

    for [x, pixel] in np.array(encoded):
        x_sum += x

    # print("x_sum is ", x_sum)

    return [np.array(encoded).astype(np.uint32), less_img]


def rl_decode(encoded, shape):
    decoded = []

    for rl in encoded:
        r, p = rl[0], rl[1]
        decoded.extend([p]*r)

    dimg = np.array(decoded).reshape(shape)
    return dimg

def test_run_length_rgb(fpath):
    # Open the image
    preimg = np.asarray(Image.open(fpath))

    img0 = preimg[:,:,0]
    img1 = preimg[:,:,1]
    img2 = preimg[:,:,2]

    # Encode the image
    [encoded0, less_img0] = rl_encode(img0)
    [encoded1, less_img1] = rl_encode(img1)
    [encoded2, less_img2] = rl_encode(img2)

    # plt.imshow(less_img, cmap="gray")
    # plt.show()

    # print("encoded0 is ", encoded0)

    # Encoding sure does reduce the size
    print("sizes")
    print(sys.getsizeof(img0)+sys.getsizeof(img1)+sys.getsizeof(img2))
    print(sys.getsizeof(encoded0)+ sys.getsizeof(encoded1)+ sys.getsizeof(encoded2))

    # print("encoded dimensions: {} by {}".format(encoded[0], encoded[1]))

    # Decode the encoding to another image
    shape = preimg.shape
    dimg0 = rl_decode(encoded0, (shape[0], shape[1]))
    dimg1 = rl_decode(encoded1, (shape[0], shape[1]))
    dimg2 = rl_decode(encoded2, (shape[0], shape[1]))

    # Calculate the standard deviation betweeen decoding and the original image
    diff_sum = 0
    for i in range(len(less_img0)):
        for j in range(len(less_img0[0])):
            diff_sum += (less_img0[i][j]-dimg0[i][j])**2

    std_dev = (diff_sum/(i*j))**(1/2)
    print("standard deviation is {}".format(std_dev))

    # Create a figure to show the original, grayscale and decoded images
    fig, axs = plt.subplots(1, 2)

    dimg = np.array([[[0 for i in range(3)] for j in range(len(dimg0[0]))] for k in range(len(dimg0))])

    # print(dimg0)

    dimg[:,:,0] = dimg0
    dimg[:,:,1] = dimg1
    dimg[:,:,2] = dimg2

    axs[0].imshow(preimg)
    axs[0].set_title("Original image")
    axs[1].imshow(dimg)
    axs[1].set_title("Decoded image")
    plt.show()



def test_run_length(fpath):

    # Open the image
    preimg = np.asarray(Image.open(fpath))

    img = get_greyscale_image(preimg)

    # Encode the image
    [encoded, less_img] = rl_encode(img)

    # plt.imshow(less_img, cmap="gray")
    # plt.show()

    # print("encoded is ", encoded)

    # Encoding sure does reduce the size
    print("sizes")
    print(sys.getsizeof(img))
    print(sys.getsizeof(encoded))

    # print("encoded dimensions: {} by {}".format(encoded[0], encoded[1]))

    # Decode the encoding to another image
    shape = img.shape
    dimg = rl_decode(encoded, (shape[0], shape[1]))

    # Calculate the standard deviation betweeen decoding and the original image
    diff_sum = 0
    for i in range(len(less_img)):
        for j in range(len(less_img[0])):
            diff_sum += (less_img[i][j]-dimg[i][j])**2

    std_dev = (diff_sum/(i*j))**(1/2)
    print("standard deviation is {}".format(std_dev))

    # Create a figure to show the original, grayscale and decoded images
    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(preimg)
    axs[0].set_title("Original image")
    axs[1].imshow(less_img, cmap="gray")
    axs[1].set_title("Gray image")
    axs[2].imshow(dimg, cmap="gray")
    axs[2].set_title("Decoded image")
    plt.show()


if __name__ == '__main__':
    # test_greyscale("monkey.gif")
    # test()
    # test_rgb("lena.gif")

    # Create a parser
    parser = argparse.ArgumentParser(description="Program to perform compression \
                                     and decompression on images")

    # Define the command-line arguments
    parser.add_argument("-f", "--file", help="Image file path")
    parser.add_argument("-o", "--option",
                        help="Compression-decompression technique")
    parser.add_argument("-t", "--type", help="Type of the image")

    # Parse the command line
    args = parser.parse_args()

    # Access the argument values
    file_path = args.file
    opt = args.option
    typ = args.type

    err_msg = "Correct usage is: python3 compression.py -f <file> -o <option> -t <type>\n\
              Replace the <file> with the address of the image file\n\
              Replace the <option> with compression technique you want to be performed\n\
              Replace <type> with the image type you want to use in the program\n\
              The options are fractal-rgb or fractal-grey or run-length-grey or run-length-rgb\n\
              Type the line below to the command line to get more info:\n\
              python3 compression.py -h"

    if (not file_path or not opt):
        print(err_msg)

    elif (opt == "fractal-rgb"):
        test_rgb(file_path, typ)

    elif (opt == "fractal-grey"):
        test_greyscale(file_path, typ)
    elif (opt == "run-length-grey"):
        test_run_length(file_path)
    elif (opt == "run-length-rgb"):
        test_run_length_rgb(file_path)

    else:
        print(err_msg)
