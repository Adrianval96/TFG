#! /usr/bin/env bash

# ae_neurons = [15, 23, 29, 35, 45]
# ae_learning_rates = [0.1, 0.3, 0.9]
# ae_corruption_levels = [0.1, 0.3, 0.9]
# ae_training_epochs = [1000, 3000, 5000]

# mlp_neurons = [13, 19, 25, 35]
# mlp_learning_rates = [0.1, 0.3, 0.9]
# mlp_momentum_factors = [0.1, 0.3, 0.9]
# mlp_training_epochs = [1000, 3000, 5000]


if [ ! -d /SCRATCH/japaca1/results/lluvia_categorizada ]; then
	mkdir -p /SCRATCH/japaca1/results/lluvia_categorizada
fi;

for i in 15 23 29 35 45; do
    for j in 0.1 0.3 0.9; do
        for k in 0.1 0.3 0.9; do
	    for l in 1000 3000 5000; do
                for m in 13 19 25 35; do
                    for n in 0.1 0.3 0.9; do
                        for o in 0.1 0.3 0.9; do
                            for p in 1000 3000 5000; do
                                if [ ! -f /SCRATCH/japaca1/results/lluvia_categorizada/Prediction\ $i\,$j\,$k\,$l\_$m\,$n\,$o\,$p.csv ]; then
                                    if (( `qstat | wc -l` > 4990 )); then
                                        sleep 3600
                                    fi;
                                    qsub run_experiment.sh $i $j $k $l $m $n $o $p lluvia_categorizada;
                                    echo "Running experiment $i $j $k $l $m $n $o $p lluvia_categorizada";

				else 
                                    echo "Experiment $i $j $k $l $m $n $o $p was already done";
                                fi;

done;
done;
done;
done;
done;
done;
done;
done;
