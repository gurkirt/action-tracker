module load gcc/8.2.0 python_gpu/3.10.4 libjpeg-turbo eth_proxy; unset PYTHONPATH
WANDB_API_KEY=45113a54ba7487c773500127750d17e755fdd527
source ~/envirs/pt113/bin/activate

source experiments/mov_mot_trainval.sh

outdir='/cluster/work/cvl/gusingh/data/tracking/experiments/gtr_train4'

python train_net.py --num-gpus 4 --eval-only --config-file configs/GTR_MOT_FPN.yaml SOLVER.IMS_PER_BATCH 4 INPUT_DIR $TARGET_DIR OUTPUT_DIR $outdir MODEL.WEIGHTS /cluster/work/cvl/gusingh/data/tracking/experiments/gtr_train4/model_final.pth