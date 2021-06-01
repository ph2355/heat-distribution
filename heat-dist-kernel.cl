__kernel void heat_distribution(__global float *plate,	
						__global float *plateNew,
						 int width,
                         int height,
                         float tile_width,
                         float tile_height)						
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0); 
    int j = get_global_id(1); 
    
    if(i > 0 && j > 0 && i < height - 1 && j < width - 1) {
        // these are used, so the equation below is shorter
        int W = width;
        int H = height;
        float w = tile_width;
        float h = tile_height;
        __global float *T = plate;

        plateNew[i * W + j] = 1.0/2.0 * (((T[(i + 1) * W + j] + T[(i - 1) * W + j]) / (1 + (w * w / (h * h)))) + ((T[i * W + j + 1] + T[i * W + j - 1]) / (1 + (h * h / (w * w)))));
    }
}

__kernel void heat_distribution_lmem(__global float *plate,	
						__global float *plateNew,
						 int width,
                         int height,
                         float tile_width,
                         float tile_height,
                        __local float *lmem)						
{
 
    // Get the index of the current element to be processed
    int i = get_global_id(0); 
    int j = get_global_id(1); 

    int li = get_local_id(0);
    int lj = get_local_id(1);

    int lsize_h = get_local_size(0);
    int lsize_w = get_local_size(1);

    int lmem_h = get_local_size(0) + 2;
    int lmem_w = get_local_size(1) + 2;

    if (i < height && j < width) {
        lmem[(li + 1) * lmem_w + lj + 1] = plate[i * width + j];

        if (li == 0 && i > 0) {
            lmem[(li + 1 - 1) * lmem_w + lj + 1] = plate[(i - 1) * width + j];
            
            if (lj == 0 && j > 0)
                lmem[(li + 1 - 1) * lmem_w + lj + 1 - 1] = plate[(i - 1) * width + j - 1];

            if (lj == lsize_w - 1 && j < width - 1)
                lmem[(li + 1 - 1) * lmem_w + lj + 1 + 1] = plate[(i - 1) * width + j + 1];
        }

        if (li == lsize_h - 1 && i < height - 1) {
            lmem[(li + 1 + 1) * lmem_w + lj + 1] = plate[(i + 1) * width + j];

            if (lj == 0 && j > 0)
                lmem[(li + 1 + 1) * lmem_w + lj + 1 - 1] = plate[(i + 1) * width + j - 1];

            if (lj == lsize_w - 1 && j < width - 1)
                lmem[(li + 1 + 1) * lmem_w + lj + 1 + 1] = plate[(i + 1) * width + j + 1];
        }

        if (lj == 0 && j > 0) {
            lmem[(li + 1) * lmem_w + lj + 1 - 1] = plate[i * width + j - 1];
        } 
        if (lj == lsize_w - 1 && j < width - 1) {
            lmem[(li + 1) * lmem_w + lj + 1 + 1] = plate[i * width + j + 1];
        }
    }

    // if(i == 0 && j == 0) {
    //     for (int c = i; c < i + lsize_h; c++) {
    //         for (int c2 = j; c2 < j + lsize_w; c2++) {
    //             printf("%f ", plate[c * width + c2]);
    //         }
    //         printf("\n");
    //     }

    //     printf("\n");
    //     printf("result\n");

    //     for (int c = 0; c < lmem_h; c++) {
    //         for (int c2 = 0; c2 < lmem_w; c2++) {
    //             printf("%f ", lmem[c * lmem_w + c2]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
       
    // }


    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(i > 0 && j > 0 && i < height - 1 && j < width - 1) {
        // these are used, so the equation below is shorter
        int W = lmem_w;
        int H = lmem_h;
        // int W = width;
        // int H = height;
        float w = tile_width;
        float h = tile_height;
        __global float *T = plate;

        // plateNew[i * W + j] = 1.0/2.0 * (((T[(i + 1) * W + j] + T[(i - 1) * W + j]) / (1 + (w * w / (h * h)))) + ((T[i * W + j + 1] + T[i * W + j - 1]) / (1 + (h * h / (w * w)))));
        plateNew[i * width + j] = 1.0/2.0 * (((lmem[(li + 1 + 1) * W + lj + 1] + lmem[(li - 1 + 1) * W + lj + 1]) / (1 + (w * w / (h * h)))) + ((lmem[(li + 1) * W + lj + 1 + 1] + lmem[(li + 1) * W + lj - 1 + 1]) / (1 + (h * h / (w * w)))));
    }
}

