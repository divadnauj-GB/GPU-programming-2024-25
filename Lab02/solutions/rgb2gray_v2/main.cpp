// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/img.h"


int main(void) {

    RGBImage *Input_image;
    GrayImage *Output_image;

    Input_image = readPPM("sample_640_426.ppm");

    Output_image = createPGM(Input_image->width, Input_image->height);

    kernel_rgb2gray(Input_image, Output_image);

    writePGM("sample_640_426_gray.pgm",Output_image);

    destroyPPM(Input_image);
    destroyPGM(Output_image);

    printf("\nProgramm Finished!\n");
    return 0;
}