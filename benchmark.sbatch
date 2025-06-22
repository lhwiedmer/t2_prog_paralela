#!/bin/bash

#SBATCH --job-name=lcs_benchmark # Nome do job para identificação
#SBATCH --nodes=2                      # Solicita exatamente 2 nós
#SBATCH --ntasks=12                    # Número total de tarefas (2 nós * 6 tarefas/nó)
#SBATCH --output=lcs_benchmark_%j.out  # Arquivo de saída padrão
#SBATCH --error=lcs_benchmark_%j.err   # Arquivo de erro padrão

#==============================================================================
# SCRIPT DE SUBMISSÃO SLURM
# Configurado para 2 nós com 6 núcleos cada.
#==============================================================================

echo "============================================================"
echo "Iniciando job Slurm para 2 nós"
echo "Job ID: $SLURM_JOB_ID"
echo "Nós alocados: $SLURM_JOB_NODELIST"
echo "Tasks por nó: $SLURM_NTASKS_PER_NODE"
echo "Total de tasks: $SLURM_NTASKS"
echo "Diretório de trabalho: $(pwd)"
echo "============================================================"
date

# Carregue os módulos MPI necessários (se houver).
# Esta parte é específica do seu cluster.
# Exemplo:
# module load openmpi

# Concede permissão de execução ao script de lógica
chmod +x ./roda_testes.sh

# Executa o script de lógica do benchmark.
# O Slurm e o mpirun irão gerenciar a distribuição dos processos
# entre os 2 nós alocados.
bash ./roda_testes.sh

echo "============================================================"
echo "Job Slurm concluído."
date
echo "============================================================"