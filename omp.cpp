# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>


#define WIDTH 1000
#define HEIGHT 1000
#define TILE_WIDTH 1
#define TILE_HEIGHT 1
#define M (WIDTH / TILE_WIDTH + (WIDTH % TILE_WIDTH == 0 ? 0 : 1))
#define N (HEIGHT / TILE_HEIGHT + (HEIGHT % TILE_HEIGHT == 0 ? 0 : 1))
#define N_ITERATIONS 20

double u[M][N];
double w[M][N];

void printPlate() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2lf ", w[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{

    int width_m = TILE_WIDTH;
    int width_n = TILE_HEIGHT;
    int i;
    int iterations;
    int j;
    double wtime;


#pragma omp parallel shared ( w ) private ( i, j )
    {
#pragma omp for
        for (i = 1; i < M - 1; i++)
        {
            w[i][0] = 100.0;
        }
#pragma omp for
        for (i = 1; i < M - 1; i++)
        {
            w[i][N - 1] = 100.0;
        }
#pragma omp for
        for (j = 0; j < N; j++)
        {
            w[M - 1][j] = 0.0;
        }
#pragma omp for
        for (j = 0; j < N; j++)
        {
            w[0][j] = 100.0;
        }


    }
#pragma omp parallel shared ( mean, w ) private ( i, j )
    {
#pragma omp for
        for (i = 1; i < M - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                w[i][j] = 0;
            }
        }
    }

    printf("Zacetno stanje:\n");
    printPlate();

    iterations = 0;
    wtime = omp_get_wtime();
    int iter = N_ITERATIONS;

    while (iterations <= iter)
    {
# pragma omp parallel shared ( u, w ) private ( i, j )
        {
# pragma omp for
            for (i = 0; i < M; i++)
            {
                for (j = 0; j < N; j++)
                {
                    u[i][j] = w[i][j];
                }
            }

# pragma omp for
            for (i = 1; i < M - 1; i++)
            {
                for (j = 1; j < N - 1; j++)
                {
                    w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;

                    w[i][j] = 0.5 * (((u[i + 1][j] + u[i - 1][j]) / (1 + (width_m * width_m / width_n * width_n))) + ((u[i][j + 1] + u[i][j - 1]) / (1 + (width_n * width_n / width_m * width_m))));
                }
            }
        }
        iterations++;
    }
    wtime = omp_get_wtime() - wtime;

    printf("Koncno stanje:\n");
    printPlate();

    printf("Cas poteka %f\n", wtime);

    printf("\n");

    return 0;

# undef M
# undef N
}