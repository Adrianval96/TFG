# Las lineas que comienzan por #$ son indicaciones al sistema de colas.
#$ -cwd              # Ejecutar en directorio actual.
#$ -l h_vmem=4g      # Memoria requerida. 
#$ -l h_rt=08:00:00  # Tiempo requerido (hh:mm:ss).
#$ -q gpus -l gpu=1


# ae_neurons = [15, 23, 29, 35, 45]
# ae_learning_rates = [0.1, 0.3, 0.9]
# ae_corruption_levels = [0.1, 0.3, 0.9]
# ae_training_epochs = [1000, 3000, 5000]

# mlp_neurons = [13, 19, 25, 35]
# mlp_learning_rates = [0.1, 0.3, 0.9]
# mlp_momentum_factors = [0.1, 0.3, 0.9]
# mlp_training_epochs = [1000, 3000, 5000]

source /home/japaca1/rigel35/bin/activate

cd /home/japaca1/lluvia_categorizada
/home/japaca1/rigel35/bin/python SimpleAutoEncoder_cat.py $@


