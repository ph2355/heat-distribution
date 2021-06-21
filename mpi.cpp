#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"


int calc_ncols_from_rank(int rank, int size, int grid_size)
{
    int ncols;

    ncols = grid_size / size;       
    if ((grid_size % size) != 0) {  
        if (rank == size - 1)
            ncols += grid_size % size; 
    }

    return ncols;
}

void init_plate(double** plate, int grid_size, int local_nrows, int local_ncols, int rank, int size) {
    double boundary_mean = ((grid_size - 2) * 100.0 * 2 + (grid_size - 2) * 100.0) / (double)((2 * grid_size) + (2 * grid_size) - 4);
    for (int i = 0; i < local_nrows; i++) {
        for (int j = 1; j < local_ncols + 1; j++) {
            if (i == 0)
                plate[i][j] = 100.0;
            else if (i == local_nrows - 1)
                plate[i][j] = 0.0;
            else if ((rank == 0) && j == 1)                  
                plate[i][j] = 100.0;
            else if ((rank == size - 1) && j == local_ncols) 
                plate[i][j] = 100.0;
            else
                plate[i][j] = boundary_mean;
        }
    }
}

int main(int argc, char* argv[])
{
    int grid_size = 16;
    int max_iters = 20;
    int master = 0;
    int start_col, end_col; 
    int rank;              
    int size;              
    int tag = 0;  
    int width_n = 1;
    int width_m = 1;	

    MPI_Status status;     
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int left = (rank == master) ? (rank + size - 1) : (rank - 1);
    int right = (rank + 1) % size;
    int local_nrows = grid_size;
    int local_ncols = calc_ncols_from_rank(rank, size, grid_size);

    double** plate_prev = (double**)malloc(sizeof(double*) * local_nrows);
    for (int i = 0; i < local_nrows; i++) {
        plate_prev[i] = (double*)malloc(sizeof(double) * (local_ncols + 2));
    }
    double** plate_now = (double**)malloc(sizeof(double*) * local_nrows);
    for (int i = 0; i < local_nrows; i++) {
        plate_now[i] = (double*)malloc(sizeof(double) * (local_ncols + 2));
    }
    double* sendbuf = (double*)malloc(sizeof(double) * local_nrows);
    double* recvbuf = (double*)malloc(sizeof(double) * local_nrows);
    int remote_ncols = calc_ncols_from_rank(size - 1, size, grid_size);
    double* printbuf = (double*)malloc(sizeof(double) * (remote_ncols + 2));
    init_plate(plate_now, grid_size, local_nrows, local_ncols, rank, size);

    double start = omp_get_wtime();

    for (int iter = 0; iter < max_iters; iter++) {


        /* pošlji levo, dobi z desne */
        for (int i = 0; i < local_nrows; i++)
            sendbuf[i] = plate_now[i][1];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, left, tag,
            recvbuf, local_nrows, MPI_DOUBLE, right, tag,
            MPI_COMM_WORLD, &status);
        for (int i = 0; i < local_nrows; i++)
            plate_now[i][local_ncols + 1] = recvbuf[i];

        /* pošlji desno, dobi z leve */
        for (int i = 0; i < local_nrows; i++)
            sendbuf[i] = plate_now[i][local_ncols];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, right, tag,
            recvbuf, local_nrows, MPI_DOUBLE, left, tag,
            MPI_COMM_WORLD, &status);
        for (int i = 0; i < local_nrows; i++)
            plate_now[i][0] = recvbuf[i];


        for (int i = 0; i < local_nrows; i++) {
            for (int j = 0; j < local_ncols + 2; j++) {
                plate_prev[i][j] = plate_now[i][j];
            }
        }

        for (int i = 1; i < local_nrows - 1; i++) {
            if (rank == 0) {
                start_col = 2;
                end_col = local_ncols;
            }
            else if (rank == size - 1) {
                start_col = 1;
                end_col = local_ncols - 1;
            }
            else {
                start_col = 1;
                end_col = local_ncols;
            }
            for (int j = start_col; j < end_col + 1; j++) {
                plate_now[i][j] =  0.5 * (((plate_prev[i + 1][j] + plate_prev[i - 1][j]) / (1 + (width_m * width_m / width_n * width_n))) + ((plate_prev[i][j + 1] + plate_prev[i][j - 1]) / (1 + (width_n * width_n / width_m * width_m))));
            }
        }

    }


    for (int i = 0; i < local_nrows; i++) {
        if (rank == 0) {
            for (int j = 1; j < local_ncols + 1; j++) {
                printf("%6.2f ", plate_now[i][j]);
            }
            for (int kk = 1; kk < size; kk++) { /* loop over other ranks */
                remote_ncols = calc_ncols_from_rank(kk, size, grid_size);
                MPI_Recv(printbuf, remote_ncols + 2, MPI_DOUBLE, kk, tag, MPI_COMM_WORLD, &status);
                for (int j = 1; j < remote_ncols + 1; j++) {
                    printf("%6.2f ", printbuf[j]);
                }
            }
            printf("\n");
        }
        else {
            MPI_Send(plate_now[i], local_ncols + 2, MPI_DOUBLE, master, tag, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    double end = omp_get_wtime();

    printf("GPU execution time %.2f\n", end - start);

    return EXIT_SUCCESS;
}
