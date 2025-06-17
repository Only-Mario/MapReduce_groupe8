#!/bin/bash
INPUT="input.txt"
MAX_ITER=25
THRESHOLD=0.001
DAMPING=0.85

cp "$INPUT" iteration0.txt
echo "Démarrage du calcul PageRank..."

for i in $(seq 1 $MAX_ITER)
do
    echo "Itération $i/$MAX_ITER"
    cat iteration$((i-1)).txt | python mapper.py | sort | python reducer.py $DAMPING > iteration$i.txt
    
    # Vérification de convergence (simplifiée)
    if [ $i -gt 1 ]; then
        DIFF=$(python -c "
import sys
try:
    with open('iteration$((i-1)).txt') as f1, open('iteration$i.txt') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        total_diff = 0
        for l1, l2 in zip(lines1, lines2):
            try:
                rank1 = float(l1.split()[1])
                rank2 = float(l2.split()[1])
                total_diff += abs(rank1 - rank2)
            except:
                pass
        print(total_diff)
except:
    print(1.0)
")
        if (( $(echo "$DIFF < $THRESHOLD" | bc -l) )); then
            echo "Convergence atteinte à l'itération $i"
            break
        fi
    fi
done

echo "Calcul terminé."
