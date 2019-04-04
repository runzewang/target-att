#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=gpu1,optimizer_including=cudnn,traceback.limit=100,lib.cnmem=0.9,floatX='float32' CUDA_LAUNCH_BLOCKING=1 allow_gc=False
export PYTHONPATH=$PYTHONPATH:../
#cd $PBS_O_WORKDIR
python -u -m train.train > train.log 2>&1 &



