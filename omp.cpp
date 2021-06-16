#define _USE_MATH_DEFINES
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<string>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<time.h>
using namespace std;



inline double sq(double a) {
    return a * a;
}

void init_plate(double** plate, int size, double** new_plate) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // plate[i][j] = 0;
            if (j == 0)
                plate[i][j] = 100.0;
            else if (j == size - 1)
                plate[i][j] = 100.0;
            else if (i == 0)                  
                plate[i][j] = 100.0;
            else if (i == size - 1) 
                plate[i][j] = 0.0;
            else
                plate[i][j] = 0;
        }
    }


    // for (int i = 0; i < size; i++) {
    //     plate[i][0] = sq(cos(i * M_PI / double(size)));
    //     plate[i][size - 1] = sq(sin(i * M_PI / double(size)));
    // }


    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            new_plate[i][j] = plate[i][j];
        }
    }
}

void plate_to_file(double** plate, int size) {
    char filename[50];
    sprintf(filename, "map_omp_%d.txt", size);
    ofstream fout(filename);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fout << i << " " << j << " " << plate[i][j] << endl;
        }fout << endl;
    }

    fout.close();
}

void printPlate(double** plate, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.2lf ", plate[i][j]);
        }
        printf("\n");
    }
}


int main(int argc, char* argv[])
{
    const int plate_size = 16; //atof(argv[0]); //velikost plate
    const int nthreads = 8; //atof(argv[1]); //�tevilo threadov
    clock_t starting_time;
    starting_time = clock();
    double avg = 0;
    const double kappa = 1;


    //  omp_set_num_threads(8);

    //inizializacija tabel
    double** start_plate = new double* [plate_size];
    for (int i = 0; i < plate_size; i++) {
        start_plate[i] = new double[plate_size];
    }

    double** new_plate = new double* [plate_size];
    for (int i = 0; i < plate_size; i++) {
        new_plate[i] = new double[plate_size];
    }

    double dx = M_PI / plate_size;
    const double dt = sq(dx) / (8 * kappa);
    const double time = 0.5 * sq(M_PI) / kappa;
    const double nsteps = 20; //time / dt;

    init_plate(start_plate, plate_size, new_plate);


    for (int tt = 0; tt < nsteps; tt++) {

#pragma omp parallel for num_threads(nthreads)   

        for (int ii = 1; ii < plate_size - 1; ii++) {
            for (int j = 1; j < plate_size - 1; j++) {
                new_plate[ii][j] = start_plate[ii][j] + kappa * dt * (start_plate[ii - 1][j] + start_plate[ii + 1][j] + start_plate[ii][j - 1] + start_plate[ii][j + 1] - 4 * start_plate[ii][j]) / sq(dx);
            }
        }
#pragma omp parallel for num_threads(nthreads)
        for (int i = 1; i < plate_size - 1; i++) {
            new_plate[0][i] = start_plate[0][i] + kappa * dt * (start_plate[plate_size - 1][i] + start_plate[1][i] + start_plate[0][i - 1] + start_plate[0][i + 1] - 4 * start_plate[0][i]) / sq(dx);
        }
#pragma omp parallel for num_threads(nthreads)
        for (int i = 1; i < plate_size - 1; i++) {
            new_plate[plate_size - 1][i] = start_plate[plate_size - 1][i] + kappa * dt * (start_plate[plate_size - 2][i] + start_plate[0][i] + start_plate[plate_size - 1][i - 1] + start_plate[plate_size - 1][i + 1] - 4 * start_plate[plate_size - 1][i]) / sq(dx);
        }

#pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < plate_size; i++) {
            for (int j = 0; j < plate_size; j++) {
                start_plate[i][j] = new_plate[i][j];
            }
        }

    }



    starting_time = clock() - starting_time;
    cout << "Porabljen cas " << float(starting_time) / CLOCKS_PER_SEC << endl;

    plate_to_file(start_plate, plate_size);
    printPlate(start_plate, plate_size);
}