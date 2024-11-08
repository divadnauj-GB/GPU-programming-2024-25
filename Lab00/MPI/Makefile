###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-11.6

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=mpic++ #mpic++ for cpp files
CC_FLAGS= -pthread
CC_LIBS=

##########################################################

### GPU architecture
SM_ARCH=86

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch=sm_${SM_ARCH}
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = .

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = .

##########################################################

## Make variables ##

# Target executable name:
EXE = run_test
# Object files:

CFILES   = $(wildcard *.c)
CPPFILES = $(wildcard *.cpp)
OBJFILES = $(CFILES:.c=.out) $(CPPFILES:.cpp=.out)

##########################################################

## Compile ##

all: $(OBJFILES) 


$(SRC_DIR)/%.out: $(SRC_DIR)/%.o 
	$(CC) -o $@ -fopenmp $<
	

$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp 
	#$(CC) -c -o $@ -fopenmp $<
	$(CC) $(CC_FLAGS) -c $< -o $@

$(SRC_DIR)/%.o : $(SRC_DIR)/%.c 
	$(CC) -c -o $@ -fopenmp $<
	
clean:
	rm -rf *.out

