#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

// O tipo de dado e as funções de leitura, alocação e liberação permanecem os mesmos.
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

mtype ** allocateScoreMatrix(int sizeA, int sizeB) {
    int i;
    mtype ** scoreMatrix = (mtype **) malloc((sizeB + 1) * sizeof(mtype *));
    for (i = 0; i < (sizeB + 1); i++) {
        // Aloca toda a linha, mesmo que o processo só calcule algumas colunas.
        // Isso simplifica a indexação. A coleta de resultados será otimizada.
        scoreMatrix[i] = (mtype *) calloc((sizeA + 1), sizeof(mtype));
    }
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
    // Garante que não ultrapasse os limites das sequências
    int i_end = (i_start + blockSize > sizeB + 1) ? sizeB + 1 : i_start + blockSize;
    int j_end = (j_start + blockSize > sizeA + 1) ? sizeA + 1 : j_start + blockSize;

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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0;
    int blockSize = 0;

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Uso: %s <block_size> [fileA] [fileB]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        blockSize = atoi(argv[1]);
        char* fileA = (argc > 2) ? argv[2] : "fileA.in";
        char* fileB = (argc > 3) ? argv[3] : "fileB.in";

        seqA = read_seq(fileA);
        seqB = read_seq(fileB);
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
        printf("Executando com %d processos.\n", num_procs);
        printf("Tamanho do bloco: %d\n", blockSize);
        printf("SeqA: %d, SeqB: %d\n", sizeA, sizeB);
    }
    
    // Broadcast dos tamanhos e do blockSize para todos os processos
    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Alocação das sequências nos outros processos
    if (rank != 0) {
        seqA = (char *) malloc((sizeA + 1) * sizeof(char));
        seqB = (char *) malloc((sizeB + 1) * sizeof(char));
    }
    // Broadcast das sequências para todos
    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Cada processo aloca e inicializa sua própria matriz de score
    mtype ** scoreMatrix = allocateScoreMatrix(sizeA, sizeB);
    initScoreMatrix(scoreMatrix, sizeA, sizeB);

    // --- PONTO 1: Cálculo prévio dos blocos para cada rank ---
    int bi = (sizeB + blockSize - 1) / blockSize;
    int bj = (sizeA + blockSize - 1) / blockSize;

    // Estrutura para guardar as coordenadas dos blocos de cada rank
    int *my_blocks_count = (int*) calloc(bi + bj, sizeof(int)); // Contagem de blocos por anti-diagonal
    int **my_blocks_coords = (int**) malloc((bi + bj) * sizeof(int*)); // Coordenadas (j_block) dos blocos

    for (int d = 0; d < bi + bj -1; ++d) {
        my_blocks_coords[d] = (int*) malloc(bj * sizeof(int));
    }

    for (int d = 0; d < bi + bj -1; ++d) {
        for (int i_block = 0; i_block <= d; ++i_block) {
            int j_block = d - i_block;
            if (i_block < bi && j_block < bj) {
                if (j_block % num_procs == rank) {
                    my_blocks_coords[d][my_blocks_count[d]++] = j_block;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // --- PONTO 3: Uso de comunicação não bloqueante ---
    mtype *col_buffer = (mtype*) malloc(blockSize * sizeof(mtype));
    MPI_Request recv_req = MPI_REQUEST_NULL;
    MPI_Request send_req = MPI_REQUEST_NULL;

    // Loop pelas anti-diagonais (wavefront)
    for (int d = 0; d < bi + bj -1; ++d) {
        // Itera apenas sobre os blocos que este rank possui nesta anti-diagonal
        for(int k = 0; k < my_blocks_count[d]; ++k) {
            int j_block = my_blocks_coords[d][k];
            int i_block = d - j_block;

            // 1. Receber dependência da esquerda (se não for a primeira coluna de blocos)
            if (j_block > 0) {
                int source_rank = (j_block - 1) % num_procs;
                // Posta um recebimento não bloqueante
                MPI_Irecv(col_buffer, blockSize, MPI_UNSIGNED_SHORT, source_rank, d, MPI_COMM_WORLD, &recv_req);

                // Espera a comunicação ser completada antes de usar o buffer
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
                
                // Copia a coluna recebida para a fronteira da matriz local
                int j_target_col = j_block * blockSize;
                int i_start_copy = 1 + i_block * blockSize;
                for(int i_copy = 0; i_copy < blockSize; i_copy++) {
                    if ((i_start_copy + i_copy) <= sizeB) {
                        scoreMatrix[i_start_copy + i_copy][j_target_col] = col_buffer[i_copy];
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
                int i_start_send = 1 + i_block * blockSize;
                for(int i_send = 0; i_send < blockSize; i_send++) {
                    if ((i_start_send + i_send) <= sizeB) {
                       col_buffer[i_send] = scoreMatrix[i_start_send + i_send][j_source_col];
                    } else {
                       col_buffer[i_send] = 0; // Padding se necessário
                    }
                }
                // Posta um envio não bloqueante. O processo pode continuar
                // enquanto o dado é enviado em background.
                MPI_Isend(col_buffer, blockSize, MPI_UNSIGNED_SHORT, dest_rank, d, MPI_COMM_WORLD, &send_req);
                // É importante esperar que o send termine antes de reutilizar o buffer
                // ou de sair do programa. Uma barreira no final ou um MPI_Waitall
                // garante isso.
                MPI_Request_free(&send_req); // Libera o handle da requisição
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // --- PONTO 2: Coleta eficiente do score e da matriz ---
    // Apenas o processo root montará a matriz final.
    // Os outros processos enviam apenas as colunas que calcularam.

    if (rank == 0) {
        // O processo 0 já tem suas próprias colunas calculadas.
        // Agora, ele recebe as colunas dos outros processos.
        mtype* col_gather_buffer = (mtype*) malloc((sizeB + 1) * sizeof(mtype));

        for (int source_rank = 1; source_rank < num_procs; ++source_rank) {
            // Itera sobre todas as colunas de blocos que pertencem ao source_rank
            for (int j_block = source_rank; j_block < bj; j_block += num_procs) {
                int j_start = j_block * blockSize + 1;
                int j_end = ((j_block + 1) * blockSize > sizeA) ? sizeA : (j_block + 1) * blockSize;

                // Recebe cada coluna do bloco
                for (int j = j_start; j <= j_end; ++j) {
                    MPI_Recv(col_gather_buffer, sizeB + 1, MPI_UNSIGNED_SHORT, source_rank, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // Copia a coluna recebida para a matriz final
                    for (int i = 1; i <= sizeB; ++i) {
                        scoreMatrix[i][j] = col_gather_buffer[i];
                    }
                }
            }
        }
        free(col_gather_buffer);

    } else {
        // Todos os outros processos enviam suas colunas de responsabilidade para o root.
        mtype* col_gather_buffer = (mtype*) malloc((sizeB + 1) * sizeof(mtype));
        
        for (int j_block = rank; j_block < bj; j_block += num_procs) {
            int j_start = j_block * blockSize + 1;
            int j_end = ((j_block + 1) * blockSize > sizeA) ? sizeA : (j_block + 1) * blockSize;
            
            // Envia cada coluna que calculou
            for (int j = j_start; j <= j_end; ++j) {
                // Prepara o buffer da coluna
                for (int i = 1; i <= sizeB; ++i) {
                    col_gather_buffer[i] = scoreMatrix[i][j];
                }
                MPI_Send(col_gather_buffer, sizeB + 1, MPI_UNSIGNED_SHORT, 0, j, MPI_COMM_WORLD);
            }
        }
        free(col_gather_buffer);
    }
    
    // O processo root imprime o resultado final. O score está na última célula.
    if (rank == 0) {
        printf("\nTempo de computação: %f segundos\n", end_time - start_time);
        // O score final é calculado e está em scoreMatrix[sizeB][sizeA]
        printf("\nScore Final: %d\n", scoreMatrix[sizeB][sizeA]);
        
        #ifdef DEBUGMATRIX
            printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
        #endif
    }

    // Liberação de memória e finalização do MPI
    free(col_buffer);
    for (int d = 0; d < bi + bj - 1; ++d) {
        free(my_blocks_coords[d]);
    }
    free(my_blocks_coords);
    free(my_blocks_count);
    free(seqA);
    free(seqB);
    freeScoreMatrix(scoreMatrix, sizeB);
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}