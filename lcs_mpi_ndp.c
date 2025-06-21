#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> // Inclusão da biblioteca MPI

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

typedef unsigned short mtype;

// As funções read_seq, allocateScoreMatrix, initScoreMatrix, freeScoreMatrix
// e processaBloco permanecem exatamente as mesmas.
#pragma region Funções Utilitárias (sem alterações)

char* read_seq(char *fname) {
    FILE *fseq = NULL;
    long size = 0;
    char *seq = NULL;
    int i = 0;
    fseq = fopen(fname, "rt");
    if (fseq == NULL ) {
        printf("Error reading file %s\n", fname);
        exit(1);
    }
    fseek(fseq, 0L, SEEK_END);
    size = ftell(fseq);
    rewind(fseq);
    seq = (char *) calloc(size + 1, sizeof(char));
    if (seq == NULL ) {
        printf("Erro allocating memory for sequence %s.\n", fname);
        exit(1);
    }
    while (!feof(fseq)) {
        seq[i] = fgetc(fseq);
        if ((seq[i] != '\n') && (seq[i] != EOF))
            i++;
    }
    seq[i] = '\0';
    fclose(fseq);
    return seq;
}

mtype ** allocateScoreMatrix(int sizeA, int sizeB) {
    int i;
    mtype ** scoreMatrix = (mtype **) malloc((sizeB + 1) * sizeof(mtype *));
    for (i = 0; i < (sizeB + 1); i++)
        scoreMatrix[i] = (mtype *) malloc((sizeA + 1) * sizeof(mtype));
    return scoreMatrix;
}

void initScoreMatrix(mtype ** scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    for (j = 0; j < (sizeA + 1); j++)
        scoreMatrix[0][j] = 0;
    for (i = 1; i < (sizeB + 1); i++)
        scoreMatrix[i][0] = 0;
}

void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
    int i;
    for (i = 0; i < (sizeB + 1); i++)
        free(scoreMatrix[i]);
    free(scoreMatrix);
}

void processaBloco(mtype** scoreMatrix, int sizeA, int sizeB, int i_block, int j_block, int blockSize, const char* seqA, const char* seqB) {
    int i_start = 1 + i_block * blockSize;
    int j_start = 1 + j_block * blockSize;
    int i_end = (i_start + blockSize < sizeB + 1) ? i_start + blockSize : sizeB + 1;
    int j_end = (j_start + blockSize < sizeA + 1) ? j_start + blockSize : sizeA + 1;

    for (int i = i_start; i < i_end; ++i) {
        for (int j = j_start; j < j_end; ++j) {
            if (seqB[i - 1] == seqA[j - 1]) {
                scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
            } else {
                scoreMatrix[i][j] = max(scoreMatrix[i - 1][j], scoreMatrix[i][j - 1]);
            }
        }
    }
}

#pragma endregion

// A função LCS_MPI permanece a mesma, pois o cálculo distribuído não muda.
mtype LCS_MPI(mtype ** scoreMatrix, int sizeA, int sizeB, char * seqA, char *seqB, int blockSize, int rank, int num_procs) {
    int bi = (sizeB + blockSize -1) / blockSize;
    int bj = (sizeA + blockSize -1) / blockSize;

    mtype *col_buffer = (mtype*) malloc(blockSize * sizeof(mtype));

    for (int d = 0; d < bi + bj; ++d) {
        for (int i_block = 0; i_block <= d; ++i_block) {
            int j_block = d - i_block;
            if (i_block < bi && j_block < bj) {
                if (j_block % num_procs == rank) {
                    if (j_block > 0) {
                        int source_rank = (j_block - 1) % num_procs;
                        MPI_Recv(col_buffer, blockSize, MPI_UNSIGNED_SHORT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        int j_target_col = j_block * blockSize;
                        int i_start = 1 + i_block * blockSize;
                        for(int k=0; k<blockSize; k++) {
                            if ((i_start + k) <= sizeB) {
                                scoreMatrix[i_start + k][j_target_col] = col_buffer[k];
                            }
                        }
                    }
                    processaBloco(scoreMatrix, sizeA, sizeB, i_block, j_block, blockSize, seqA, seqB);
                    if (j_block < bj - 1) {
                        int dest_rank = (j_block + 1) % num_procs;
                        int j_source_col = (j_block + 1) * blockSize;
                        int i_start = 1 + i_block * blockSize;
                         for(int k=0; k<blockSize; k++) {
                            if ((i_start + k) <= sizeB) {
                                col_buffer[k] = scoreMatrix[i_start + k][j_source_col];
                            }
                        }
                        MPI_Send(col_buffer, blockSize, MPI_UNSIGNED_SHORT, dest_rank, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
    
    free(col_buffer);
    // Retorna o score local. O valor correto estará no processo que possui a última célula.
    return scoreMatrix[sizeB][sizeA];
}


int main(int argc, char ** argv) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0;
    int blockSize = 32;

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <block_size> [fileA] [fileB]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        blockSize = atoi(argv[1]);
        char* fileA = (argc > 2) ? argv[2] : "fileA.in";
        char* fileB = (argc > 3) ? argv[3] : "fileB.in";

        seqA = read_seq(fileA);
        seqB = read_seq(fileB);
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
        printf("Running with %d processes.\n", num_procs);
        printf("Block size: %d\n", blockSize);
        printf("SeqA: %d, SeqB: %d\n", sizeA, sizeB);
    }
    
    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        seqA = (char *) malloc((sizeA + 1) * sizeof(char));
        seqB = (char *) malloc((sizeB + 1) * sizeof(char));
    }
    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    mtype ** scoreMatrix = allocateScoreMatrix(sizeA, sizeB);
    initScoreMatrix(scoreMatrix, sizeA, sizeB);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    LCS_MPI(scoreMatrix, sizeA, sizeB, seqA, seqB, blockSize, rank, num_procs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // === OBTENÇÃO APENAS DO SCORE FINAL (SEM MONTAR A MATRIZ) ===
    mtype final_score = 0;

    // Calcula qual processo possui o resultado final.
    // É o processo responsável pela última coluna de blocos.
    int last_j_block = (sizeA > 0) ? (sizeA - 1) / blockSize : 0;
    int owner_rank = last_j_block % num_procs;

    if (rank == 0) {
        if (owner_rank == 0) {
            // Se o processo root (0) é o dono, ele já tem o score.
            final_score = scoreMatrix[sizeB][sizeA];
        } else {
            // Senão, o processo root recebe o score do processo dono.
            MPI_Recv(&final_score, 1, MPI_UNSIGNED_SHORT, owner_rank, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        printf("\nComputation time: %f seconds\n", end_time - start_time);
        printf("\nFinal Score: %d\n", final_score);
        
    } else {
        // Se este processo (não-root) for o dono do resultado, ele o envia para o root.
        if (rank == owner_rank) {
            mtype local_score = scoreMatrix[sizeB][sizeA];
            MPI_Send(&local_score, 1, MPI_UNSIGNED_SHORT, 0, 99, MPI_COMM_WORLD);
        }
    }
    
    // Liberação de memória e finalização do MPI
    free(seqA);
    free(seqB);
    freeScoreMatrix(scoreMatrix, sizeB);
    MPI_Finalize();

    return EXIT_SUCCESS;
}