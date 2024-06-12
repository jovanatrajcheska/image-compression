import streamlit as st
import cv2
from PIL import Image
import numpy as np


def compress_jpeg(path, quality):
    img = cv2.imread(path)
    result, compressed = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return compressed


def compress_png(path, compression_level):
    image = Image.open(path)
    image.save("compressed.png", "PNG", optimize=True, compression_level=compression_level)
    with open("compressed.png", "rb") as f:
        compressed_image = np.array(Image.open(f))
    return compressed_image


def compress_webp(image_path, quality):
    image = cv2.imread(image_path)
    result, compressed_image = cv2.imencode('.webp', image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    compressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    return compressed_image


def compress_kmeans(image_path, k):
    image = cv2.imread(image_path)
    Z = image.reshape((-1, 3))

    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    compressed_image = res.reshape((image.shape))

    return compressed_image

def rle_encode(image):
    pixels = image.flatten()
    encoded_pixels = []
    prev_pixel = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoded_pixels.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1

    encoded_pixels.append((prev_pixel, count))
    return encoded_pixels


def rle_decode(encoded_pixels, shape):
    decoded_pixels = []
    for value, count in encoded_pixels:
        decoded_pixels.extend([value] * count)
    return np.array(decoded_pixels).reshape(shape)


def compress_rle(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    encoded_pixels = rle_encode(image)
    compressed_image = rle_decode(encoded_pixels, image.shape)
    return compressed_image


st.title("Demo App")

uploaded_file = st.file_uploader("Choose an image from your local machine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = uploaded_file.name
    image.save(image_path)

    st.image(image, caption='Resource', use_column_width=True)

    algorithm = st.selectbox("Choose Compression Algorithm",
                             ["JPEG", "PNG", "WebP", "K-means", "RLE"])

    if algorithm == "JPEG":
        quality = st.slider("Choose quality for the result image", 0, 100, 60)
        compressed = compress_jpeg(image_path, quality)
        compressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        st.image(compressed, channels="BGR", caption='Result', use_column_width=True)

    elif algorithm == "PNG":
        compression_level = st.slider("Choose level", 0, 9, 3)
        compressed = compress_png(image_path, compression_level)
        st.image(compressed, caption='Result', use_column_width=True)

    elif algorithm == "WebP":
        quality = st.slider("Choose quality for the result image", 0, 100, 80)
        compressed = compress_webp(image_path, quality)
        st.image(compressed, channels="BGR", caption='Result', use_column_width=True)

    elif algorithm == "K-means":
        k = st.slider("Number of Colors (K)", 1, 30, 3)
        compressed = compress_kmeans(image_path, k)
        st.image(compressed, caption='Result', use_column_width=True)

    elif algorithm == "RLE":
        compressed = compress_rle(image_path)
        st.image(compressed, caption='Result', use_column_width=True)
