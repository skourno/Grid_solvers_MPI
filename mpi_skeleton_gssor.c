#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"

//Computational Kernels

void Jacobi(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max)
{
    int i, j;
    for (i = X_min; i < X_max; i++)
        for (j = Y_min; j < Y_max; j++)
            u_current[i][j] = (u_previous[i - 1][j] + u_previous[i + 1][j] + u_previous[i][j - 1] + u_previous[i][j + 1]) / 4.0;
}

void GaussSeidel(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega)
{
    int i, j;
    for (i = X_min; i < X_max; i++)
        for (j = Y_min; j < Y_max; j++)
            u_current[i][j] = u_previous[i][j] + (u_current[i - 1][j] + u_previous[i + 1][j] + u_current[i][j - 1] + u_previous[i][j + 1] - 4 * u_previous[i][j]) * omega / 4.0;
}

void RedSOR(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega)
{
    int i, j;
    for (i = X_min; i < X_max; i++)
        for (j = Y_min; j < Y_max; j++)
            if ((i + j) % 2 == 0)
                u_current[i][j] = u_previous[i][j] + (omega / 4.0) * (u_previous[i - 1][j] + u_previous[i + 1][j] + u_previous[i][j - 1] + u_previous[i][j + 1] - 4 * u_previous[i][j]);
}

void BlackSOR(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega)
{
    int i, j;
    for (i = X_min; i < X_max; i++)
        for (j = Y_min; j < Y_max; j++)
            if ((i + j) % 2 == 1)
                u_current[i][j] = u_previous[i][j] + (omega / 4.0) * (u_current[i - 1][j] + u_current[i + 1][j] + u_current[i][j - 1] + u_current[i][j + 1] - 4 * u_previous[i][j]);
}


int main(int argc, char ** argv)
{
    int rank, size;
    int global[2], local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int i, j, t;
    int global_converged = 0, converged = 0; //flags for convergence, global and per process
    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory
    double omega;           //relaxation factor - useless for Jacobi

    struct timeval tts, ttf, tcs, tcf; //Timers: total-tts,ttf, computation-tcs,tcf
    double ttotal = 0, tcomp = 0, total_time, comp_time;

    double ** U, ** u_current, ** u_previous, ** swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc != 5)
        {
            fprintf(stderr, "Usage: mpirun .... ./exec X Y Px Py \n");
            exit(-1);
        }
    else
        {
            global[0] = atoi(argv[1]);
            global[1] = atoi(argv[2]);
            grid[0] = atoi(argv[3]);
            grid[1] = atoi(argv[4]);
        }

    //----Create 2D-cartesian communicator----//
    //----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2] = {0, 0};    //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator

    MPI_Cart_create(MPI_COMM_WORLD, 2, grid, periods, 0, &CART_COMM); //communicator creation
    MPI_Cart_coords(CART_COMM, rank, 2, rank_grid);                 //rank mapping on the new communicator

    MPI_Barrier(CART_COMM);
    MPI_Barrier(MPI_COMM_WORLD);
    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//

    for (i = 0; i < 2; i++)
        {
            if (global[i] % grid[i] == 0)
                {
                    local[i] = global[i] / grid[i];
                    global_padded[i] = global[i];
                }
            else
                {
                    local[i] = (global[i] / grid[i]) + 1;
                    global_padded[i] = local[i] * grid[i];
                }
        }

    //Initialization of omega
    omega = 2.0 / (1 + sin(3.14 / global[0]));


    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    if (rank == 0)
        {
            U = allocate2d(global_padded[0], global_padded[1]);
            init2d(U, global[0], global[1]);
        }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous = allocate2d(local[0] + 2, local[1] + 2);
    u_current = allocate2d(local[0] + 2, local[1] + 2);

    //----Distribute global 2D-domain from rank 0 to all processes----//

    //----Appropriate datatypes are defined here----//
    /*****The usage of datatypes is optional*****/

    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0], local[1], global_padded[1], MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0], local[1], local[1] + 2, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &local_block);
    MPI_Type_commit(&local_block);

    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int * scatteroffset, * scattercounts;
    if (rank == 0)
        {
            scatteroffset = (int*)malloc(size * sizeof(int));
            scattercounts = (int*)malloc(size * sizeof(int));
            for (i = 0; i < grid[0]; i++)
                for (j = 0; j < grid[1]; j++)
                    {
                        scattercounts[i * grid[1] + j] = 1;
                        scatteroffset[i * grid[1] + j] = (local[0] * local[1] * grid[1] * i + local[1] * j);
                    }
        }


    //----Rank 0 scatters the global matrix----//

    double * initaddr;
    if (rank == 0)
        initaddr = &(U[0][0]);

    MPI_Scatterv(initaddr, scattercounts, scatteroffset, global_block, &(u_previous[1][1]), 1, local_block, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initaddr, scattercounts, scatteroffset, global_block, &(u_current[1][1]), 1, local_block, 0, MPI_COMM_WORLD);

    if (rank == 0)
        free2d(U, global_padded[0], global_padded[1]);

    //----Define datatypes or allocate buffers for message passing----//
    MPI_Datatype mat_row;
    MPI_Type_vector(1, local[1], 0, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &mat_row);
    MPI_Type_commit(&mat_row);

    MPI_Datatype mat_column;
    MPI_Type_vector(local[0], 1, local[1] + 2, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &mat_column);
    MPI_Type_commit(&mat_column);

    //************************************//


    //----Find the 4 neighbors with which a process exchanges messages----//

    //*************TODO*******************//
    int north, south, east, west;
    /*Make sure you handle non-existing
        neighbors appropriately*/
    //Init to -1
    north = -1;
    south = -1;
    east = -1;
    west = -1;

    //Try to get north Process
    if (rank_grid[0] - 1 >= 0)
        {
            int npos[2] = {rank_grid[0] - 1, rank_grid[1]};
            MPI_Cart_rank(CART_COMM, npos , &north);
        }

    //Try to get south Process
    if (rank_grid[0] + 1 <= grid[0] - 1)
        {
            int npos[2] = {rank_grid[0] + 1, rank_grid[1]};
            MPI_Cart_rank(CART_COMM, npos, &south);
        }

    //Try to get east Process
    if (rank_grid[1] + 1 <= grid[1] - 1)
        {
            int npos[2] = {rank_grid[0], rank_grid[1] + 1};
            MPI_Cart_rank(CART_COMM, npos, &east);
        }

    //Try to get west Process
    if (rank_grid[1] - 1 >= 0)
        {
            int npos[2] = {rank_grid[0], rank_grid[1] - 1};
            MPI_Cart_rank(CART_COMM, npos, &west);
        }


    //************************************//


    //---Define the iteration ranges per process-----//
    //*************TODO*******************//

    int i_min, i_max, j_min, j_max;

    /*Three types of ranges:
        -internal processes
        -boundary processes
        -boundary processes and padded global array
    */

    //Init Values for internal processes
    i_min = 1;
    i_max = local[0] + 1;

    j_min = 1;
    j_max = local[1] + 1;


    //Fix stuff according to neighbors found
    //This Should fix Boundary Processes
    if (north == -1)
        {
            i_min += 1;
        }
    if (south == -1)
        {
            i_max -= 1;
        }
    if (west == -1)
        {
            j_min += 1;
        }
    if (east == -1)
        {
            j_max -= 1;
        }

    //Fix Padded Bounds
    if (rank_grid[0] == grid[0] - 1)
        {
            i_max -= global_padded[0] - global[0];
        }

    if (rank_grid[1] == grid[1] - 1)
        {
            j_max -= global_padded[1] - global[1];
        }



    printf("Process (%d, %d) R: %2d Neighbors: N: %2d S: %2d E: %2d W: %2d Working Size: %d x %d Imin %d, Imax %d, Jmin %d, Jmax %d\n", \
           rank_grid[0], rank_grid[1], rank,  north, south, east, west, local[0] + 2, local[1] + 2, i_min, i_max, j_min, j_max);

    //************************************//



    //Define MPI_Requests for all interactions
    MPI_Request mpi_reqns_1, mpi_reqns_2;
    MPI_Request mpi_reqew_1, mpi_reqew_2;
    MPI_Status mpistatus;
    //----Computational core----//
    gettimeofday(&tts, NULL); //Get Starting Time
#   ifdef TEST_CONV
    for (t = 0; t < T && !global_converged; t++)
        {
#   endif
#   ifndef TEST_CONV
#   undef T
#   define T 65536
            for (t = 0; t < T; t++)
                {
#   endif

                    //Swap Buffers
                    swap = u_previous;
                    u_previous = u_current;
                    u_current = swap;
                    //Communicate
                    /*
                    Message Tags:
                    Transfer Top Row 50
                    Transfer Bottom Row 60
                    Transfer East Column 70
                    Transfer West Column 80
                    */
                    //Invoke send and recv async requests for anything that can be transfered
                    //North South interaction
                    if (north != -1 || south != -1)
                        {
                            if (north != -1)
                                {
                                    //Send top row to north
                                    MPI_Isend(&u_previous[1][1], 1, mat_row, north, 50, MPI_COMM_WORLD, &mpi_reqns_1);
                                    //Receive lower row from north
                                    MPI_Irecv(&u_previous[0][1], 1, mat_row, north, 60, MPI_COMM_WORLD, &mpi_reqns_2);
                                }
                            if (south != -1)
                                {
                                    //Send bottom row to south
                                    MPI_Isend(&u_previous[i_max - 1][1], 1, mat_row, south, 60, MPI_COMM_WORLD, &mpi_reqns_2);
                                    //Receive top row from south
                                    MPI_Irecv(&u_previous[i_max][1], 1, mat_row, south, 50, MPI_COMM_WORLD, &mpi_reqns_1);
                                }
                            //Wait for completion
                            MPI_Wait(&mpi_reqns_1, &mpistatus);
                            MPI_Wait(&mpi_reqns_2, &mpistatus);
                        }
                    //East West Interaction
                    if (east != -1 || west != -1)
                        {
                            if (east != -1)
                                {
                                    //Send Right Column to east
                                    MPI_Isend(&u_previous[i_min][j_max - 1], 1, mat_column, east, 70, MPI_COMM_WORLD, &mpi_reqew_1);
                                    //Receive
                                    MPI_Irecv(&u_previous[i_min][j_max], 1, mat_column, east, 80, MPI_COMM_WORLD, &mpi_reqew_2);
                                }
                            if (west != -1)
                                {
                                    MPI_Isend(&u_previous[i_min][j_min], 1, mat_column, west, 80, MPI_COMM_WORLD, &mpi_reqew_2);
                                    //Receive left column from west
                                    MPI_Irecv(&u_previous[i_min][0], 1, mat_column, west, 70, MPI_COMM_WORLD, &mpi_reqew_1);
                                }
                            //Wait for completion
                            MPI_Wait(&mpi_reqew_1, &mpistatus);
                            MPI_Wait(&mpi_reqew_2, &mpistatus);
                        }

                    //Start Computation
                    /*Add appropriate timers for computation*/
                    gettimeofday(&tcs, NULL);

                    //Computational Kernels
                    //Modified GSSOR Kernel
                    int i, j;
                    for (i = i_min; i < i_max; i++)
                        {
                            for (j = j_min; j < j_max; j++)
                                {
                                    //Receives before Calculations
                                    if (i == i_min && j = j_min && north != -1)
                                        {
                                            //Receive Updated Elements from Upper Process
                                            MPI_Recv(&u_current[i_min - 1][j_min], 1, mat_row, north, 60, MPI_COMM_WORLD);
                                            break;
                                        }

                                    if (j == j_min && west != -1)
                                        {
                                            //Receive Updated Element from left process
                                            MPI_Recv(&u_current[i][0], 1, MPI_DOUBLE, west, 70, MPI_COMM_WORLD);
                                        }

                                    u_current[i][j] = u_previous[i][j] + (u_current[i - 1][j] + u_previous[i + 1][j] + \
                                                                          u_current[i][j - 1] + u_previous[i][j + 1] - \
                                                                          4 * u_previous[i][j]) * omega / 4.0;
                                    //Sends after Calculation
                                    if (j == j_max - 1 && east != -1)
                                        {
                                            //Send Updated last element to the right process
                                            MPI_Isend(&u_current[i][j], 1, MPI_DOUBLE, east, 70, MPI_COMM_WORLD,  &mpi_reqew_2);
                                            //MPI_Wait(&mpi_reqew_2, &mpistatus);
                                        }

                                    if (i == i_max - 1 && j == j_max - 1 && south != -1)
                                        {
                                            //Send Updated Elements from Upper Process
                                            MPI_Isend(&u_current[i_max-1][j_min], 1, mat_row, south, 60, MPI_COMM_WORLD, &mpi_reqns_1);
                                            //printf("Proc %d sending bottom row update\n", rank);
                                            //MPI_Wait(&mpi_reqns_1, &mpistatus);
                                        }
                                }
                        }


                    gettimeofday(&tcf, NULL);
                    //Calculate Computation Time,  Average
                    tcomp = (tcomp + (tcf.tv_sec - tcs.tv_sec) + (tcf.tv_usec - tcs.tv_usec) * 0.000001) / 2.;


#               ifdef TEST_CONV
                    if (t % C == 0)
                        {
                            //*************TODO**************//
                            /*Test convergence*/
                            converged = converge(&(u_previous[1]), &(u_current[1]), local[0], local[1]);
                            if (converged)
                                printf("Process: %d Converged\n", rank);
                            MPI_Allreduce(&converged, &global_converged, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
                        }
#               endif

                    //************************************//

                }
            printf("Rank: %d,  Done Computing\n", rank);
            gettimeofday(&ttf, NULL);

            ttotal = (ttf.tv_sec - tts.tv_sec) + (ttf.tv_usec - tts.tv_usec) * 0.000001;

            MPI_Barrier(MPI_COMM_WORLD); //Make sure all processes have finished computation

            //The following reduction is for the time sum
            MPI_Reduce(&ttotal, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&tcomp, &comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);



            //----Rank 0 gathers local matrices back to the global matrix----//
            if (rank == 0)
                {
                    U = allocate2d(global_padded[0], global_padded[1]);
                    initaddr = &(U[0][0]);
                    printf("Value of T : %d\n", T);
                }


            //All Processes send data back to rank0,  rank0 receives
            //Use Gatherv Command
            MPI_Gatherv(&(u_current[1][1]), 1, local_block, initaddr, scattercounts, scatteroffset, global_block, 0, MPI_COMM_WORLD);


            //************************************//

            //----Printing results----//

            //**************TODO: Change "Jacobi" to "GaussSeidelSOR" or "RedBlackSOR" for appropriate printing****************//
            if (rank == 0)
                {

#           ifdef PRINT_RESULTS
                    char * s = malloc(50 * sizeof(char));

#           ifdef JACOBI
                    printf("Jacobi X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n", \
                           global[0], global[1], grid[0], grid[1], t, comp_time, total_time, U[global[0] / 2][global[1] / 2]);
                    sprintf(s, "res%sMPI_%dx%d_%dx%d", "Jacobi", global[0], global[1], grid[0], grid[1]);
#           endif

#           ifdef GSSOR
                    printf("GaussSeidel X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n", \
                           global[0], global[1], grid[0], grid[1], t, comp_time, total_time, U[global[0] / 2][global[1] / 2]);
                    sprintf(s, "res%sMPI_%dx%d_%dx%d", "GaussSeidel", global[0], global[1], grid[0], grid[1]);
#           endif

#           ifdef REDBLACK
                    printf("RedBlackSOR X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n", \
                           global[0], global[1], grid[0], grid[1], t, comp_time, total_time, U[global[0] / 2][global[1] / 2]);
                    sprintf(s, "res%sMPI_%dx%d_%dx%d", "RedBlackSOR", global[0], global[1], grid[0], grid[1]);
#           endif

                    fprint2d(s, U, global[0], global[1]);
                    free(s);
#           endif

                }
            MPI_Finalize();
            return 0;

        }
