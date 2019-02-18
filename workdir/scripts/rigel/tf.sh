#$ -cwd
#$ -l h_vmem=4g

source activate openai
conda list

/home/advagi/anaconda3/envs/openai/bin/python tf.py
