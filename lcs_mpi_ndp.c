#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

typedef unsigned short mtype;

// Estrutura e funções utilitárias permanecem as mesmas
#pragma region Structs e Funções Utilitárias
typedef struct {
    mtype **data;
    int *global_j_map;
    int num_local_j_blocks;
    int sizeA;
    int sizeB;
    int blockSize;
    int rank;
    int num_procs;
} DistMatrix;

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
DistMatrix* create_dist_matrix(int sizeA, int sizeB, int blockSize, int rank, int num_procs) {
    DistMatrix* mat = (DistMatrix*)malloc(sizeof(DistMatrix));
    mat->sizeA = sizeA; mat->sizeB = sizeB; mat->blockSize = blockSize; mat->rank = rank; mat->num_procs = num_procs;
    int bj = (sizeA + blockSize - 1) / blockSize;
    mat->num_local_j_blocks = 0;
    for (int jb = 0; jb < bj; ++jb) if (jb % num_procs == rank) mat->num_local_j_blocks++;
    mat->global_j_map = (int*)malloc(mat->num_local_j_blocks * sizeof(int));
    int local_jb_idx = 0;
    for (int jb = 0; jb < bj; ++jb) if (jb % num_procs == rank) mat->global_j_map[local_jb_idx++] = jb;
    int local_width = mat->num_local_j_blocks * blockSize + 1;
    mat->data = (mtype **)malloc((sizeB + 1) * sizeof(mtype *));
    for (int i = 0; i < (sizeB + 1); i++) mat->data[i] = (mtype *)calloc(local_width, sizeof(mtype));
    return mat;
}
void free_dist_matrix(DistMatrix *mat) {
    for (int i = 0; i < (mat->sizeB + 1); i++) free(mat->data[i]);
    free(mat->data); free(mat->global_j_map); free(mat);
}
int get_local_j_block_idx(DistMatrix* mat, int global_jb) {
    for (int i = 0; i < mat->num_local_j_blocks; ++i) if (mat->global_j_map[i] == global_jb) return i;
    return -1;
}
void processa_bloco_local(DistMatrix* mat, int i_block, int local_j_block_idx, const char* seqA, const char* seqB) {
    int global_j_block = mat->global_j_map[local_j_block_idx];
    int i_start = 1 + i_block * mat->blockSize;
    int j_start_global = 1 + global_j_block * mat->blockSize;
    int j_start_local = 1 + local_j_block_idx * mat->blockSize;
    int i_end = (i_start + mat->blockSize < mat->sizeB + 1) ? i_start + mat->blockSize : mat->sizeB + 1;
    int j_end_local = j_start_local + mat->blockSize;
    for (int i = i_start; i < i_end; ++i) {
        for (int j_local = j_start_local; j_local < j_end_local; ++j_local) {
            int j_global = j_start_global + (j_local - j_start_local);
            if (j_global > mat->sizeA) continue;
            if (seqB[i - 1] == seqA[j_global - 1]) {
                mat->data[i][j_local] = mat->data[i - 1][j_local - 1] + 1;
            } else {
                mat->data[i][j_local] = max(mat->data[i - 1][j_local], mat->data[i][j_local - 1]);
            }
        }
    }
}
#pragma endregion

// CORREÇÃO 1: Sincronização da tag MPI na função de cálculo
void LCS_MPI_Optimized(DistMatrix* mat, const char* seqA, const char* seqB) {
    int bi = (mat->sizeB + mat->blockSize - 1) / mat->blockSize;
    int bj = (mat->sizeA + mat->blockSize - 1) / mat->blockSize;

    mtype *col_buffer = (mtype*)malloc(mat->blockSize * sizeof(mtype));

    for (int d = 0; d < bi + bj - 1; ++d) {
        for (int i_block = 0; i_block <= d; ++i_block) {
            int j_block = d - i_block;
            if (i_block < bi && j_block < bj) {
                if (j_block % mat->num_procs == mat->rank) {
                    int local_j_block_idx = get_local_j_block_idx(mat, j_block);

                    if (j_block > 0) {
                        int source_rank = (j_block - 1 + mat->num_procs) % mat->num_procs;
                        // O recebimento acontece na iteração 'd', e deve corresponder ao envio da iteração 'd-1'
                        MPI_Recv(col_buffer, mat->blockSize, MPI_UNSIGNED_SHORT, source_rank, d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        int j_target_col_local = local_j_block_idx * mat->blockSize;
                        int i_start = 1 + i_block * mat->blockSize;
                        for(int k=0; k < mat->blockSize; k++) {
                            if ((i_start + k) <= mat->sizeB) {
                                mat->data[i_start + k][j_target_col_local] = col_buffer[k];
                            }
                        }
                    }

                    processa_bloco_local(mat, i_block, local_j_block_idx, seqA, seqB);
                    
                    if (j_block < bj - 1) {
                        int dest_rank = (j_block + 1) % mat->num_procs;
                        int j_source_col_local = (local_j_block_idx + 1) * mat->blockSize;
                        int i_start = 1 + i_block * mat->blockSize;
                         for(int k=0; k<mat->blockSize; k++) {
                            if ((i_start + k) <= mat->sizeB) {
                                col_buffer[k] = mat->data[i_start + k][j_source_col_local];
                            } else {
                                col_buffer[k] = 0;
                            }
                        }
                        // O envio na iteração 'd' é para a iteração 'd+1' do processo vizinho
                        // A tag deve ser a da iteração de recebimento, ou seja, 'd+1'.
                        MPI_Send(col_buffer, mat->blockSize, MPI_UNSIGNED_SHORT, dest_rank, d + 1, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
    
    free(col_buffer);
}


int main(int argc, char ** argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0, blockSize = 32;

    if (rank == 0) {
        if (argc < 2) { fprintf(stderr, "Usage: %s <block_size> [fileA] [fileB]\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
        blockSize = atoi(argv[1]);
        char* fileA = (argc > 2) ? argv[2] : "fileA.in";
        char* fileB = (argc > 3) ? argv[3] : "fileB.in";
        seqA = read_seq(fileA); seqB = read_seq(fileB);
        sizeA = strlen(seqA); sizeB = strlen(seqB);
        printf("Running with %d processes (Optimized Memory - CORRECTED).\n", num_procs);
        printf("Block size: %d\n", blockSize); printf("SeqA: %d, SeqB: %d\n", sizeA, sizeB);
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

    DistMatrix* dist_matrix = create_dist_matrix(sizeA, sizeB, blockSize, rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    LCS_MPI_Optimized(dist_matrix, seqA, seqB);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    mtype final_score = 0;
    int last_j_block_global = (sizeA > 0) ? (sizeA - 1) / blockSize : 0;
    int owner_rank = last_j_block_global % num_procs;

    if (rank == owner_rank) {
        int last_j_block_local = get_local_j_block_idx(dist_matrix, last_j_block_global);
        int j_offset = (sizeA - 1) % blockSize;
        int last_j_local = last_j_block_local * blockSize + j_offset + 1;
        final_score = dist_matrix->data[sizeB][last_j_local];
        if (rank != 0) { MPI_Send(&final_score, 1, MPI_UNSIGNED_SHORT, 0, 99, MPI_COMM_WORLD); }
    }
    if (rank == 0) {
        if (owner_rank != 0) { MPI_Recv(&final_score, 1, MPI_UNSIGNED_SHORT, owner_rank, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
        printf("Tempo total de score%f\n", end_time - start_time);
        printf("Final Score: %d\n", final_score);
    }
    
    free(seqA); free(seqB);
    free_dist_matrix(dist_matrix);
    MPI_Finalize();
    return EXIT_SUCCESS;
}