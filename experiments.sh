export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python experiments.py