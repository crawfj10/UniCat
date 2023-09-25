# train

root='vit_rn_fusion_cat_s128x256'
#root='vit_rnt_fusion_av_s256x128'
out_path='/home/ubuntu/jenni/logs/mm/rgbn300/hpo/'
ckpt_path='/home/ubuntu/jenni/ckpts/mm/rgbn300/hpo/'
for lr in .008 .016 .032
do
    for seed in 1235 1236 1237 1238
    do
        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
	--nproc_per_node=4 --master_port 1234 train.py \
	--config_file configs/RGBN300/dc_former.yml \
	MODEL.DIST_TRAIN True SOLVER.SEED $seed SOLVER.BASE_LR $lr  \
	MODEL.USE_FUSION True MODEL.FUSION_METHOD "cat" \
	SOLVER.IMS_PER_BATCH 256 \
	OUTPUT_DIR "${out_path}seed_${seed}/${root}_bs_256_lr_${lr}" \
	CKPT_DIR "${ckpt_path}seed_${seed}/${root}_bs_256_lr_${lr}"
    done
done


#CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/Flare/dc_former.yml 
# test
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
