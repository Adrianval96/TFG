


#$ -cwd
#$ -l h_vmem=4g
#$ -l h_rt=00:30:00

source activate openai

cd /home/advagi/TFG_OpenAI/workdir/rigel/executables/python_scripts
/home/advagi/anaconda3/envs/openai/bin/python test.py
