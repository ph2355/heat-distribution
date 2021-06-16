#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 16
#define HEIGHT 16
#define TILE_WIDTH 1
#define TILE_HEIGHT 1
#define N_TILES_HORIZONTAL (WIDTH / TILE_WIDTH + (WIDTH % TILE_WIDTH == 0 ? 0 : 1))
#define N_TILES_VERTICAL (HEIGHT / TILE_HEIGHT + (HEIGHT % TILE_HEIGHT == 0 ? 0 : 1))
#define N_ITERATIONS 20

void heat_distribution_serial(float *plate, float *plateNew);
void initialize_heat_plate(float *plate);
void swap(float **plate, float **plateNew);
void printPlate(float *plate);

int main() {

    float plate[N_TILES_HORIZONTAL][N_TILES_VERTICAL];
    float plateNew[N_TILES_HORIZONTAL][N_TILES_VERTICAL];

    initialize_heat_plate((float *)plate);
    initialize_heat_plate((float *)plateNew);

    // printf("before: \n");
    // printPlate((float *)plate);

    double start = omp_get_wtime();

    heat_distribution_serial((float *)plate, (float *)plateNew);

    double end = omp_get_wtime();

    printf("serial execution time %.2f\n", end-start);

    printf("after: \n");
    printPlate((float *)plate);

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

void heat_distribution_serial(float *plate, float *plateNew) {
    // these are used, so the equation below is shorter
    int W = N_TILES_HORIZONTAL;
    int H = N_TILES_VERTICAL;
    float w = TILE_WIDTH;
    float h = TILE_HEIGHT;
    float *T = plate;

    for (int k = 0; k < N_ITERATIONS; k++) {
        for (int i = 1; i < N_TILES_VERTICAL - 1; i++) {
            for (int j = 1; j < N_TILES_HORIZONTAL - 1; j++) {
                plateNew[i * W + j] = 1.0/2.0 * (((T[(i + 1) * W + j] + T[(i - 1) * W + j]) / (1 + (w * w / (h * h)))) + ((T[i * W + j + 1] + T[i * W + j - 1]) / (1 + (h * h / (w * w)))));
            }
        }
        swap(&plate, &plateNew);
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
            printf("%6.2f ", plate[i * N_TILES_HORIZONTAL + j]);
        }
        printf("\n");
    }
}