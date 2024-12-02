# Laboratory 07

**Goals:**

- To develop parallel applications on GPUs for fast image processing algorithms
- To implement efficiently image processing algorithms based on 2-D convolution operators

## Excersice 1: Image filtering

In image processing, a kernel, convolution matrix, or mask is a small matrix used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between the kernel and the image. <https://en.wikipedia.org/wiki/Kernel_(image_processing)>

This exercise follows the similar implementation as the Exercise 4 introduced in Lab03. You can use it as starting point, or you can write your own version.

1. implement the following 2-D filtering of images both in RGB or grayscale, using diferent filters as well as different kernel sizes

|||
|-|-|
|Mean filter| $$\frac{1}{9} \cdot \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$|
|Sobel filter $G_x$| $$ \begin{pmatrix} +1 & 0 & -1 \\ +2 & 0 & -2 \\ +1 & 0 & -1 \end{pmatrix} $$|
|Sobel filter $G_y$| $$ \begin{pmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{pmatrix} $$|
|Prewitt filter $G_x$| $$ \begin{pmatrix} +1 & +1 & +1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{pmatrix} $$|
|Prewitt filter $G_y$| $$ \begin{pmatrix} +1 & 0 & -1 \\ +1 & 0 & -1 \\ +1 & 0 & -1 \end{pmatrix} $$|
|Gaussian filter $\sigma=1$| $$ \frac{1}{273} \cdot \begin{pmatrix} 1 & 4 & 7 & 4 & 1 \\ 4 & 16 & 26 & 16 & 4 \\ 7 & 26 & 41 & 26 & 7 \\ 4 & 16 & 26 & 16 & 4 \\ 1 & 4 & 7 & 4 & 1 \end{pmatrix} $$|
|Laplacian filter|$$ \begin{pmatrix} 0  & 0  & -1 & 0  & 0  \\ 0  & -1 & -2 & -1 & 0  \\ -1 & -2 & 17 & -2 & -1 \\ 0  & -1 & -2 & -1 & 0  \\ 0  & 0 & -1 & 0  & 0  \end{pmatrix} $$|
|Sharpen filter| $$ \begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix} $$|

Write the code to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory

Support different masks and different mask sizes. Evaluate performance (global memory r/w, execution time…) and compare CPU (to be coded) and GPU solutions. Consider different input sizes (32x32, SD, HD, FHD, 4K) and different mask sizes.
