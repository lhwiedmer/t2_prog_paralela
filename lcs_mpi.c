#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> // Inclusão da biblioteca MPI


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

// O tipo de dado e a leitura de sequência permanecem os mesmos
typedef unsigned short mtype;

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

// As funções de alocação, inicialização e liberação são as mesmas
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

// A função de processamento do bloco permanece a mesma
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

// A função LCS é substituída pela versão MPI
mtype LCS_MPI(mtype ** scoreMatrix, int sizeA, int sizeB, char * seqA, char *seqB, int blockSize, int rank, int num_procs) {
    int bi = (sizeB + blockSize -1) / blockSize;
    int bj = (sizeA + blockSize -1) / blockSize;

    // Buffer para comunicação da coluna de fronteira
    mtype *col_buffer = (mtype*) malloc(blockSize * sizeof(mtype));

    // Loop pelas anti-diagonais dos blocos (wavefront)
    for (int d = 0; d < bi + bj; ++d) {
        for (int i_block = 0; i_block <= d; ++i_block) {
            int j_block = d - i_block;

            if (i_block < bi && j_block < bj) {
                // Verifica se este processo é responsável por esta coluna de bloco
                if (j_block % num_procs == rank) {
                    
                    // 1. Receber dependência da esquerda (se não for a primeira coluna de blocos)
                    if (j_block > 0) {
                        int source_rank = (j_block - 1) % num_procs;
                        MPI_Recv(col_buffer, blockSize, MPI_UNSIGNED_SHORT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // Copia a coluna recebida para a fronteira da matriz local
                        int j_target_col = j_block * blockSize;
                        int i_start = 1 + i_block * blockSize;
                        for(int k=0; k<blockSize; k++) {
                            if ((i_start + k) <= sizeB) {
                                scoreMatrix[i_start + k][j_target_col] = col_buffer[k];
                            }
                        }
                    }

                    // 2. Processar o bloco
                    processaBloco(scoreMatrix, sizeA, sizeB, i_block, j_block, blockSize, seqA, seqB);

                    // 3. Enviar fronteira para a direita (se não for a última coluna de blocos)
                    if (j_block < bj - 1) {
                        int dest_rank = (j_block + 1) % num_procs;
                        
                        // Preenche o buffer com a última coluna do bloco recém-calculado
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

    // No final, cada processo tem suas colunas. O resultado final estará no canto inferior direito.
    // A montagem da matriz completa no root será feita no main.
    return scoreMatrix[sizeB][sizeA];
}

void printMatrix(char * seqA, char * seqB, mtype ** scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    printf("Score Matrix:\n");
    printf("========================================\n");
    printf("    ");
    printf("%5c   ", ' ');
    for (j = 0; j < sizeA; j++) printf("%5c   ", seqA[j]);
    printf("\n");
    for (i = 0; i < sizeB + 1; i++) {
        if (i == 0) printf("    ");
        else printf("%c   ", seqB[i - 1]);
        for (j = 0; j < sizeA + 1; j++) {
            printf("%5d   ", scoreMatrix[i][j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}


int main(int argc, char ** argv) {
    int rank, num_procs;

    // Inicialização do MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0;
    int blockSize = 32;

    if (rank == 0) {
        char* fileA = (argc > 1) ? argv[2] : "fileA.in";
        char* fileB = (argc > 2) ? argv[3] : "fileB.in";

        seqA = read_seq(fileA);
        seqB = read_seq(fileB);
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
    }
    
    // Broadcast dos tamanhos e do blockSize
    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Alocação e broadcast das sequências
    if (rank != 0) {
        seqA = (char *) malloc((sizeA + 1) * sizeof(char));
        seqB = (char *) malloc((sizeB + 1) * sizeof(char));
    }
    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Cada processo aloca e inicializa sua própria matriz
    mtype ** scoreMatrix = allocateScoreMatrix(sizeA, sizeB);
    initScoreMatrix(scoreMatrix, sizeA, sizeB);

    // Barreira para sincronizar antes de começar o cálculo
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Executa a computação principal
    LCS_MPI(scoreMatrix, sizeA, sizeB, seqA, seqB, blockSize, rank, num_procs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // === Coleta da Matriz Final no Processo Root ===
    if (rank == 0) {
        mtype **tempMatrix = allocateScoreMatrix(sizeA, sizeB);

        int bj = (sizeA + blockSize -1) / blockSize;

        // Recebe as matrizes dos outros processos e monta a final
        for (int source_rank = 1; source_rank < num_procs; ++source_rank) {
            // Recebe a matriz inteira do processo source_rank
            for(int i = 0; i <= sizeB; ++i) {
                MPI_Recv(tempMatrix[i], sizeA + 1, MPI_UNSIGNED_SHORT, source_rank, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Copia as colunas de bloco de responsabilidade do source_rank para a matriz final
            for (int j_block = source_rank; j_block < bj; j_block += num_procs) {
                int j_start = 1 + j_block * blockSize;
                int j_end = (j_start + blockSize < sizeA + 1) ? j_start + blockSize : sizeA + 1;
                for (int i = 1; i <= sizeB; ++i) {
                    for (int j = j_start; j < j_end; ++j) {
                        scoreMatrix[i][j] = tempMatrix[i][j];
                    }
                }
            }
        }
        freeScoreMatrix(tempMatrix, sizeB);

    } else {
        // Todos os outros processos enviam sua matriz para o root
        for(int i = 0; i <= sizeB; ++i) {
             MPI_Send(scoreMatrix[i], sizeA + 1, MPI_UNSIGNED_SHORT, 0, i, MPI_COMM_WORLD);
        }
    }

    double true_end_time = MPI_Wtime();
    // O processo root imprime o resultado
    if (rank == 0) {
        printf("\nTotal computation time: %f seconds\n", true_end_time - start_time);
        printf("\nComputation time: %f seconds\n", end_time - start_time);
        printf("\nFinal Score: %d\n", scoreMatrix[sizeB][sizeA]);
        
        #ifdef DEBUGMATRIX
            printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
        #endif
    }

    // Liberação de memória e finalização do MPI
    free(seqA);
    free(seqB);
    freeScoreMatrix(scoreMatrix, sizeB);
    MPI_Finalize();

    return EXIT_SUCCESS;
}