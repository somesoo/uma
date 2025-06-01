#!/bin/bash

echo 'Starting optimization loop...'

mkdir -p logs
k_start=5

for i in {1..20}
do
  echo "=== Iteration $i ==="

  # Określ wartość -w zależnie od numeru iteracji
  if [ "$i" -le 5 ]; then
    W=0
    K=3
  elif [ "$i" -le 10 ]; then
    W=1
    K=2
  elif [ "$i" -le 15 ]; then
    W=1
    K=4
  else
    W=1
    K=4
  fi

  echo "--- Running main.py (iteration $i, wildcards = $W) ---"
  python3 main.py --data_type acceptor > logs/iteration_$i.log

  echo "--- Evaluating regex performance ---"
  python3 -m testing.regex_common >> logs/iteration_$i.log

  echo "--- Updating regex set ---"
  python3 testing/update_regex.py --features_path features.csv --regex_input input_data/regex_acceptor.txt --regex_output input_data/regex_acceptor.txt >> logs/iteration_$i.log

  echo "--- Generating new regexes (wildcards = $W) ---"
  python3 src/custom_regex_generator.py -k "$K" -w "$W" -n 80 -o input_data/regex_acceptor.txt
done


echo 'Optimization finished.'