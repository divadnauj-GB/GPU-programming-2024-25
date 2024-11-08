# Laboratory 02

Debugging CUDA programs and implementing basic image processing in parallel using GPUs

## **Exercise 1: Debugging CUDA programs**

**Objective:** Use the cuda-gdb tool for debugging CUDA programs

The provided programs were developed to be executed on an Ampere GPU (i.e., `sm_86`); please make sure you change the compilation configurations to match your GPU device. In the case of Jetson Nano, you should use `sm_53`, which corresponds to a Maxwell GPU architecture.

The example programs correspond to a `matrix multiplication` and a `vector multiply accumulation`. The programs do not operate correctly. Your tasks for this laboratory consist of using any debugging strategy to find and solve the mistakes. Among the debugging strategies you may use:

1. The `cuda-dbg` tool allows you to execute every line code of the host and GPU programs step by step.
2. Code instrumentation (i.e., Use `print` functions to check parts of the code)

**Using cuda-dbg on Linux machine:**

Use Nsight Visual Studio Code Edition as introduced in the following link:
<https://developer.nvidia.com/nsight-visual-studio-code-edition>

**Using cuda-dbg on windows machine:**

Use NVIDIA nsight application for cuda debugging as introduced in the following link: <https://docs.nvidia.com/nsight-visual-studio-edition/cuda-debugger/>

## **Exercise 2: Image Processing (RGB to Gray conversion) v1**

**Objective:** The purpose of this exercise is to convert an RGB image into a grayscale image.

The input is an RGB triple of unsigned char values; your task will be to write a CUDA program to convert the RGB image into a single unsigned char grayscale intensity value. There are multiple methods for converting a color RGB image to grayscale. The following equations provide different ways of converting RGB to grayscale. The indexes `𝑖`, and `𝑗` correspond to the coordinates of an individual pixel inside the image.

- $Gray[i][j]=\frac{R[i][j]+G[i][j]+B[i][j]}{3}$
- $Gray[i][j]=0.3 \times R[i][j]+0.6\times G[i][j]+0.11\times B[i][j]$
- $Gray[i][j]=0.21\times R[i][j]+0.71\times G[i][j]+0.07\times B[i][j]$

**Support Functions** The file `imglib_v01.zip` contains a set of helper functions that allows you to handle images without using the OpenCV library. img.h and img.cpp include useful functions for reading and writing PBM and PPM. You can extend them to support more formats in different depths or implement your own better version.

**Image Format** For people who are developing their own system, the input image is stored in PPM P6 format, while the output grayscale image is stored in PPM P5 format. Students can create their own input images by exporting their images into PPM images. The easiest way to create an image is via external tools. On Unix, `bmptoppm`
converts BMP images to PPM images. (more info <https://en.wikipedia.org/wiki/Netpbm/>)

**Assignment**
Write the code to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel or kernels
- copy results from device to host
- deallocate device memory

Compare CPU (to be coded) and GPU solutions. Consider different input size (32x32, SD, HD, FHD, 4K).

## **Exercise 2: Image Processing (RGB to Gray conversion) v2**

**Objective:** Write a CUDA program to convert an image from RGB colorspace to gray color space. Modify the helper functions or create new ones to store the images as a bi-dimensional array of elements (pixels), and every pixel corresponds to a structure of three members as follows:

```cpp
struct {
    char r;
    char g;
    char b;
    } pixel;
```

**Assignment** Write the code to handle the new data structure used for the images:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel or kernels
- copy results from device to host
- deallocate device memory

Compare CPU (to be coded) and GPU solutions. Consider different input size (32x32, SD, HD, FHD, 4K).