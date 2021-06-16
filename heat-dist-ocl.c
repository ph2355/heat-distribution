#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <CL/cl.h>

#define WIDTH 1024
#define HEIGHT 1024
#define TILE_WIDTH 3
#define TILE_HEIGHT 3
#define N_TILES_HORIZONTAL (WIDTH / TILE_WIDTH + (WIDTH % TILE_WIDTH == 0 ? 0 : 1))
#define N_TILES_VERTICAL (HEIGHT / TILE_HEIGHT + (HEIGHT % TILE_HEIGHT == 0 ? 0 : 1))
#define N_ITERATIONS 10000

// openCL specific
#define WORKGROUP_SIZE	16
#define MAX_SOURCE_SIZE	16384

void heat_distribution_gpu(float *plate, float *plateNew, int optimization);
void initialize_heat_plate(float *plate);
void swap(float **plate, float **plateNew);
void printPlate(float *plate);

int main(int argc, char **argv) {

    int optimization = 0;

    float plate[N_TILES_HORIZONTAL][N_TILES_VERTICAL];
    float plateNew[N_TILES_HORIZONTAL][N_TILES_VERTICAL];

    initialize_heat_plate((float *)plate);
    initialize_heat_plate((float *)plateNew);

    // printf("before: \n");
    // printPlate((float *)plate);

    heat_distribution_gpu((float *)plate, (float *)plateNew, optimization);

    // printf("after: \n");
    // printPlate((float *)plate);

    return 0;
}

void initialize_heat_plate(float *plate) {
    for (int i = 0; i < N_TILES_VERTICAL; i++) {
        for (int j = 0; j < N_TILES_HORIZONTAL; j++) {
            if (j == 0)
                plate[i * N_TILES_HORIZONTAL + j] = 100;        // initialize left side of plate
            else if (j == N_TILES_HORIZONTAL - 1)
                plate[i * N_TILES_HORIZONTAL + j] = 100;        // initialize right side of plate
            else if (i == 0)
                plate[i * N_TILES_HORIZONTAL + j] = 100;        // initialize top side of plate
            else if (i == N_TILES_VERTICAL - 1)
                plate[i * N_TILES_HORIZONTAL + j] = 0;          // initialize bottom side of plate
            else 
                plate[i * N_TILES_HORIZONTAL + j] = 0;
            
        }
    }

    for (int i = 0; i < N_TILES_VERTICAL; i++) {
        // initialize left side of plate
        plate[i * N_TILES_HORIZONTAL + 0] = 100;
        
        // initialize right side of plate
        plate[i * N_TILES_HORIZONTAL + N_TILES_HORIZONTAL - 1] = 100;
    }

    for (int i = 0; i < N_TILES_HORIZONTAL; i++) {
        // initialize top side of plate
        plate[i] = 100;
        
        // initialize bottom side of plate
        plate[(N_TILES_VERTICAL - 1) * N_TILES_HORIZONTAL + i] = 0;
    }
}

void swap(float **plate, float **plateNew) {
    float *temp = *plate;
    *plate = *plateNew;
    *plateNew = temp;
}

void printPlate(float *plate) {
    for (int i = 0; i < N_TILES_VERTICAL; i++) {
        for (int j = 0; j < N_TILES_HORIZONTAL; j++) {
            printf("%f ", plate[i * N_TILES_HORIZONTAL + j]);
        }
        printf("\n");
    }
}

void heat_distribution_gpu(float *plate, float *plateNew, int optimization) {

	// OpenCL initialization
	// read kernel source
	FILE* fp;
	char* source_str;
	char fileName[100];
	size_t source_size;

	fp = fopen("heat-dist-kernel.cl", "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

    // Get platforms
    cl_uint num_platforms;
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

	// create a context
	cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);

	// create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

	// create and build a program
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &clStatus);
	clStatus = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char *build_log;
	clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return;
    }

    double start = omp_get_wtime();

	// divide work among threads
	int size = N_TILES_VERTICAL * N_TILES_HORIZONTAL;
	const size_t local_item_size[2] = {WORKGROUP_SIZE, WORKGROUP_SIZE};
    size_t global_item_size[2];
	global_item_size[0] = (((N_TILES_VERTICAL - 1) / WORKGROUP_SIZE) + 1) * WORKGROUP_SIZE;
    global_item_size[1] = (((N_TILES_HORIZONTAL - 1) / WORKGROUP_SIZE) + 1) * WORKGROUP_SIZE;
    int width = N_TILES_HORIZONTAL;
    int height = N_TILES_VERTICAL;
    float tile_width = TILE_WIDTH;
    float tile_height = TILE_HEIGHT;

	// alocate device memory
	cl_mem plate_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
								 	sizeof(float) * size, plate, &clStatus);

    cl_mem plateNew_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
								 	sizeof(float) * size, plateNew, &clStatus);

    char *progName = "heat_distribution";
    if (optimization) 
        progName = "heat_distribution_lmem";

    cl_kernel kernel1 = clCreateKernel(program, progName, &clStatus);
    clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&plate_gpu);
    clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&plateNew_gpu);
    clSetKernelArg(kernel1, 2, sizeof(cl_int), (void*)&width);
    clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*)&height);
    clSetKernelArg(kernel1, 4, sizeof(cl_float), (void*)&tile_width);
    clSetKernelArg(kernel1, 5, sizeof(cl_float), (void*)&tile_height);
    if (optimization) {
        clSetKernelArg(kernel1, 6, (local_item_size[0] + 2) * (local_item_size[1] + 2) * sizeof(float), NULL);
    }

    cl_kernel kernel2 = clCreateKernel(program, progName, &clStatus);
    clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&plateNew_gpu);
    clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&plate_gpu);
    clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&width);
    clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*)&height);
    clSetKernelArg(kernel2, 4, sizeof(cl_float), (void*)&tile_width);
    clSetKernelArg(kernel2, 5, sizeof(cl_float), (void*)&tile_height);
    if (optimization) {
        clSetKernelArg(kernel2, 6, (local_item_size[0] + 2) * (local_item_size[1] + 2) * sizeof(float), NULL);
    }


    for (int i = 0; i < N_ITERATIONS; i++) {
        clStatus = clEnqueueNDRangeKernel(command_queue, i % 2 == 0 ? kernel1 : kernel2, 
                                            2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    }

    // copy results back
    clStatus = clEnqueueReadBuffer(command_queue, N_ITERATIONS % 2 == 0 ? plate_gpu : plateNew_gpu, CL_TRUE, 0,
                                    sizeof(float) * size, plate, 0, NULL, NULL);

	// free memory
	clStatus = clFlush(command_queue);
	clStatus = clFinish(command_queue);
	clStatus = clReleaseKernel(kernel1);
    clStatus = clReleaseKernel(kernel2);
	clStatus = clReleaseProgram(program);
	clStatus = clReleaseMemObject(plate_gpu);
	clStatus = clReleaseMemObject(plateNew_gpu);
	clStatus = clReleaseCommandQueue(command_queue);
	clStatus = clReleaseContext(context);
	
	free(devices);
	free(platforms);

    double end = omp_get_wtime();

    printf("GPU execution time %.2f\n", end-start);
}