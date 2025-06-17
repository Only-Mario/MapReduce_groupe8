#!/bin/bash
INPUT="input.txt"
MAX_ITER=30
THRESHOLD=0.001
DAMPING=0.85

cp "$INPUT" iteration0.txt
echo "D�marrage du calcul PageRank..."

for i in $(seq 1 $MAX_ITER)
do
    echo "It�ration $i/$MAX_ITER"
    cat iteration$((i-1)).txt | python mapper.py | sort | python reducer.py $DAMPING > iteration$i.txt
    
    # V�rification de convergence (simplifi�e)
    if [ $i -gt 1 ]; then
    DIFF=$(python -c "
prev = open(f'iteration{i-1}.txt').read().split()
curr = open(f'iteration{i}.txt').read().split()
diff = sum(abs(float(prev[i+1])-float(curr[i+1])) for i in range(0,len(prev),3))
print(diff)
")
    if python -c "exit(0 if $DIFF < $THRESHOLD else 1)"; then
        echo "Convergence atteinte à l'itération $i"
        break
    fi
fi
done

echo "Calcul termin�."
