# Laboratory 00

Setting up tools and boards, parallel and distributed programming, first CUDA example

## **Exercise 0: Setting up the Environment for Parallel Programming (Threads, OpenMP, and MPI)**

1. Use your laptop or laboratory computer to start a virtual machine with Ubuntu in any version (the suggested version is Ubuntu 20.04.4 Focal Fossa). You can download a ready-to-use image from [www.osboxes.org](www.osboxes.org). For example, the following link will give you the DVI file for VirtualBox; if you have a different Virtual machine engine, you need to download the right version through the webpage.

2. Uncompress the downloaded file and create a new virtual machine as suggested in this guide, with the following configurations:
    - Number of CPUs: 8
    - Ram Memory: 2048MB
3. Start the virtual machine and login using the username and password as: osboxes.org
4. Open a terminal and install the required packages and tools as follows:
    - `sudo apt-get update`
    - `sudo apt install build-essential`
    - `sudo apt install openmpi-bin openmpi-doc libopenmpi-dev`
    - `sudo apt install make`

## **Exercise 1: Introduction to Threads, OpenMP, and MPI**

The following exercises were developed to understand the differences among different parallel programming languages and tools. Log in to the virtual machine you created in the previous steps and be sure you have the directories under this directory, You will find three directories named as follows:

1. C++Threads
2. OpenMP
3. MPI

### Threads

1. Open a terminal and change the directory to the first folder C++Threads
2. Compile every file by executing the following command:
    - `g++ -std=c++11 <Myprogram>.cpp -o <MyProgramTest> -lpthread`
3. Run every program and explain why the program generates the outputs based on the source code of
every exercise
    - `./Myprogram`
4. What are the main differences of using “thread”, “pthread” and “fork process”?

### OpenMP

1. Open a terminal and change the directory to the second folder OpenMP
2. Compile every file by executing the following command:
    - `g++ <Myprogram>.cpp -o <MyProgramTest> -fopenmp`
3. Run every program and explain why the program generates the outputs based on the source code of
every exercise
    - `time ./Myprogram`
4. Comment all the #pragmas in the source files; compile again, and record the execution time. Are there
any differences? What do the #pragmas configurations do?

### MPI

1. Open a terminal and change the directory to the second folder MPI
2. Compile every file by executing the following command:
    - `mpic++ <Myprogram>.cpp -o <MyProgramTest> -fopenmp`
3. Run every program and explain why the program generates the outputs based on the source code of
every exercise.
    - `time mpirun -np 4 ./Myprogram`
4. Execute every program with a different number of processors (from 2 to 16) and record the execution
time.
5. Compile the programs again, but this time, do not include the `-fopenmp` flag
6. Run the programs and record the time again. What can you conclude from the experiments?

### Questions

- Based on the last exercises, give an explanation about the main features of Threads, pthreads, OpenMP, and MPI
- What is the difference between parallel computing and distributed computing?
- Is it possible to combine parallel computing with distributed computing? Give an alternative example where you can apply this concept.

## **Exercise 2: Setting up the NVIDIA Jetson Nano Board**

- **BOARD** (recommended):
  - Please follow the instructions reported here: <https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit>

- **HOST** (if you want CUDA on your laptop):
  - Install CUDA toolkit (11.8 recommended)
  - Optional: Setup your favorite IDE (e.g., Visual Studio, CLion)

- **Verify CUDA instalation**
  - Open a terminal (either in jetson nano or your laptop) and execute the following command:
    - `nvcc –version`
    - you should receive a message with the version of cuda installed plus the CUDA compiler driver information. In case you get an error please go to the next steps.
  - Open the file `~/.bashrc` and add the following lines at the and of it:

    ```bash
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64\   
                            ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

  - Save the file and from the terminal execute: `source ~/.bashrc`
  - Go to step 1 and assure nvcc is visible for compilation

## **Exercise 3: Run the First CUDA application**

1. Create a source file named *.cu

```c++
#include <stdio.h>
int main(void){
        printf(Hello World from Jetson CPU!\n”);
    }
```

```cpp
#include <stdio.h>
__global__ void helloFromGPU (void) {
    printf(“Hello World from Jetson GPU!\n”);
}
int main(void) {
    // hello from GPU
    printf(“Hello World from CPU!\n”);
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
```

2. Compile the following code using the nvcc compiler as follows:

    ```bash
    nvcc -arch=sm_<xx> -o main main.cu # xx measn you GPU compute capability
    ```

3. Execute the code

    ```bash
    ./main
    ```
