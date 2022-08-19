module load gcc/8.2.0 python_gpu/3.10.4 libjpeg-turbo eth_proxy; unset PYTHONPATH
WANDB_API_KEY=45113a54ba7487c773500127750d17e755fdd527
source ~/envirs/pt113/bin/activate

# source experiments/mov_mot_trainval.sh
source experiments/mov_ucf24.sh

outdir='/cluster/work/cvl/gusingh/data/tracking/experiments/gtr_train4_ucf24'


# python train_net.py --num-gpus 4 --config-file configs/GTR_UCF24_FPN.yaml \
#                 SOLVER.IMS_PER_BATCH 4 \
#                 INPUT_DIR $TARGET_DIR OUTPUT_DIR $outdir 
                

python train_net.py --num-gpus 4 --eval-only --config-file configs/GTR_UCF24_FPN.yaml \
                SOLVER.IMS_PER_BATCH 4 \
                INPUT_DIR $TARGET_DIR OUTPUT_DIR $outdir \
                MODEL.WEIGHTS ${outdir}/model_final.pth