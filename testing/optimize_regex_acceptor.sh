#!/bin/bash

echo 'Starting optimization loop...'

mkdir -p logs
k_start=5

for i in {1..40}
do
  echo "=== Iteration $i ==="

  # Określ wartość -w zależnie od numeru iteracji
  if [ "$i" -le 5 ]; then
    W=7
    K=12
  elif [ "$i" -le 10 ]; then
    W=1
    K=5
  elif [ "$i" -le 15 ]; then
    W=4
    K=10
  elif [ "$i" -le 20 ]; then
    W=4
    K=9
  elif [ "$i" -le 25 ]; then
    W=11
    K=15
  elif [ "$i" -le 30 ]; then
    W=5
    K=11
  elif [ "$i" -le 35 ]; then
    W=3
    K=8
  else
    W=2
    K=5
  fi

  echo "--- Running main.py (iteration $i, wildcards = $W) ---"
  python3 main.py --data_type acceptor > logs/iteration_$i.log

  echo "--- Evaluating regex performance ---"
  python3 -m testing.regex_common >> logs/iteration_$i.log

  echo "--- Updating regex set ---"
  python3 testing/update_regex.py --features_path features.csv --regex_input input_data/regex_acceptor.txt --regex_output input_data/regex_acceptor.txt >> logs/iteration_$i.log
  if [ "$i" -le 5 ]; then
    echo "--- Generating new regexes (wildcards = $W) ---"
    python3 src/custom_regex_generator.py -k "$K" -w "$W" -n 80 -o input_data/regex_acceptor.txt
  else
    echo "--- Generating new regexes (wildcards = $W) ---"
    python3 src/custom_regex_generator.py -k "$K" -w "$W" -n 100 -o input_data/regex_acceptor.txt
  fi
done


echo 'Optimization finished.'