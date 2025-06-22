#!/bin/bash

# =======================================================================
# SCRIPT DE LÓGICA DE BENCHMARK PARA LCS (SERIAL E MPI)
# Este script é chamado pelo script sbatch do Slurm.
# =======================================================================

echo "Iniciando script de lógica de benchmark..."
date

# --- CONFIGURAÇÃO DO BENCHMARK ---
# Defina os executáveis
GENERATOR_EXEC="./writeRandomIn"
SERIAL_EXEC="./lcs_serial"
MPI_EXEC="./lcs_mpi"

# Defina os tamanhos de entrada para testar
INPUT_SIZES=(10000 20000 30000 40000 50000) # Adicione ou altere os tamanhos aqui

# Defina o número de processadores para o teste MPI
MPI_PROCS=(1 2 4 8 10)

# Defina o número de repetições para cada teste
REPETITIONS=20

# Arquivo de saída para os resultados finais
RESULTS_FILE="resultados_benchmark.csv"

# --- FUNÇÃO PARA CÁLCULO ESTATÍSTICO (Média e Desvio Padrão) ---
# Usa 'bc -l' para matemática de ponto flutuante
# Argumento: uma lista de valores de tempo (ex: "0.1 0.2 0.3")
calculate_stats() {
    local times=($@)
    local n=${#times[@]}

    if [ $n -lt 2 ]; then
        # Não é possível calcular desvio padrão com menos de 2 amostras
        local mean=$(printf "%.6f" ${times[0]:-0})
        echo "$mean,0.000000"
        return
    fi

    # Calcula a soma
    local sum=$(printf "%s\n" "${times[@]}" | paste -sd+ - | bc -l)
    
    # Calcula a média
    local mean=$(echo "scale=10; $sum / $n" | bc -l)

    # Calcula a soma dos quadrados das diferenças da média
    local sum_sq_diff=0
    for time in "${times[@]}"; do
        local diff=$(echo "$time - $mean" | bc -l)
        local diff_sq=$(echo "$diff * $diff" | bc -l)
        sum_sq_diff=$(echo "$sum_sq_diff + $diff_sq" | bc -l)
    done

    # Calcula o desvio padrão (da amostra, n-1)
    local std_dev=$(echo "scale=10; sqrt($sum_sq_diff / ($n - 1))" | bc -l)

    # Formata a saída para CSV
    printf "%.6f,%.6f" "$mean" "$std_dev"
}

# --- PREPARAÇÃO DO ARQUIVO DE RESULTADOS ---
# Cria o cabeçalho para o arquivo de resultados CSV
echo "Tipo,Tamanho,Processadores,Metrica,Media,DesvioPadrao" > $RESULTS_FILE

# =======================================================================
# EXECUÇÃO DOS TESTES
# =======================================================================

# Loop principal sobre cada tamanho de entrada
for tam in "${INPUT_SIZES[@]}"; do

    # -----------------------------------------------------
    # Teste Serial
    # -----------------------------------------------------
    echo ">>> Iniciando testes para lcs_serial com tamanho: $tam"
    declare -a serial_total_times=() # Array para armazenar os tempos

    for i in $(seq 1 $REPETITIONS); do
        $GENERATOR_EXEC fileA.in "$tam" &>/dev/null
        $GENERATOR_EXEC fileB.in "$tam" &>/dev/null
        
        # Executa, extrai o tempo e armazena no array
        time_val=$($SERIAL_EXEC | grep 'tempo total:' | awk '{print $3}')
        serial_total_times+=($time_val)
        echo "  - Repetição $i/$REPETITIONS: $time_val s"
    done

    # Calcula as estatísticas e salva no arquivo de resultados
    stats=$(calculate_stats "${serial_total_times[@]}")
    echo "serial,$tam,1,tempo_total,$stats" >> $RESULTS_FILE
    echo ">>> Teste serial para tamanho $tam concluído. Média,DesvPad: $stats"
    echo ""

    # -----------------------------------------------------
    # Teste MPI
    # -----------------------------------------------------
    for np in "${MPI_PROCS[@]}"; do
        echo ">>> Iniciando testes para lcs_mpi com tamanho $tam e $np processadores"
        declare -a mpi_total_times=()
        declare -a mpi_seq_times=()

        for i in $(seq 1 $REPETITIONS); do
            $GENERATOR_EXEC fileA.in "$tam" &>/dev/null
            $GENERATOR_EXEC fileB.in "$tam" &>/dev/null

            # Executa o programa MPI e captura toda a saída
            output=$(mpirun -np "$np" --hostfile hostfile.txt $MPI_EXEC)
            
            # Extrai os tempos e armazena nos arrays
            total_time=$(echo "$output" | grep 'Tempo total de score:' | awk '{print $4}')
            tabel_time=$(echo "$output" | grep 'Tempo total de tabela:' | awk '{print $4}')
            
            mpi_total_times+=($total_time)
            mpi_tabel_times+=($tabel_time)
            echo "  - Repetição $i/$REPETITIONS: total=$total_time s, tab=$tabel_time s"
        done

        # Calcula e salva estatísticas para o "tempo total"
        stats_total=$(calculate_stats "${mpi_total_times[@]}")
        echo "mpi,$tam,$np,tempo_total,$stats_total" >> $RESULTS_FILE
        echo ">>> Teste MPI (tempo total) para $np procs concluído. Média,DesvPad: $stats_total"

        # Calcula e salva estatísticas para o "tempo sequencial"
        stats_tabel=$(calculate_stats "${mpi_tabel_times[@]}")
        echo "mpi,$tam,$np,tempo_sequencial,$stats_tabel" >> $RESULTS_FILE
        echo ">>> Teste MPI (tempo tab) para $np procs concluído. Média,DesvPad: $stats_tabel"
        echo ""
    done
done

# Limpa os últimos arquivos de entrada gerados
rm -f fileA.in fileB.in

echo "======================================================================="
echo "Benchmark concluído!"
echo "Os resultados foram salvos em: $RESULTS_FILE"
date
echo "======================================================================="