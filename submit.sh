#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100-SXM4:2
#SBATCH --time=6-20:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=slurm.%J.out
#cd $SLURM_SUBMIT_DIR
#cd /nlsasfs/home/sysadmin/nazgul/gpu-burn-master
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
#srun ./gpu_burn -tc -d 3600 #
#srun /bin/hostname



source  /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/k2_env.sh

export PYTHONPATH=/nlsasfs/home/nltm-pilot/msdafini/k2_icefall/icefall/icefall:$PYTHONPATH

# python3 egs/librispeech/ASR/zipformer/train.py  --world-size 4 --num-epochs 30 --start-epoch 1 --exp-dir ./exp --max-duration 400 --num-workers 32 --on-the-fly-feats True --manifest-dir /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/manifests --num-buckets 75 --bpe-model /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/lang_bpe_1200/bpe.model --train-cuts /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/manifests/tamil_train_cuts.jsonl.gz --valid-cuts /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/manifests/tamil_dev_cuts.jsonl.gz --causal 1 

python3 /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/egs/librispeech/ASR/zipformer/streaming_decode.py --epoch 29 --avg 10 --exp-dir ./exp --decoding-method greedy_search --manifest-dir /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/manifests --cut-set-name tamil_eval_cuts --bpe-model /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/lang_bpe_1200/bpe.model --causal 1 --chunk-size 32 --left-context-frames 128 --on-the-fly-feats True --use-averaged-model True --num-workers 32 --max-duration 100 --context-size 2 --num-decode-streams 500

#  python3 /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/egs/librispeech/ASR/zipformer/export.py --exp-dir ./exp --causal 1 --chunk-size 16 --left-context-frames 128 --tokens /nlsasfs/home/nltm-pilot/msdafini/k2/k2_expts/tamil/icefall/data/tamil/lang_bpe_1200/tokens.txt --epoch 29 --avg 10 --jit 1
