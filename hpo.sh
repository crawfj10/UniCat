# train

root='/home/ubuntu/jenni/logs/mm/rgbnt201/hpo/seed_0/vit_rnt_fusion_cat_s256x128'
for lr in .032
do
    for bs in 256
    do
        CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
	--nproc_per_node=4 --master_port 1236 train.py \
	--config_file configs/RGBNT201/dc_former.yml \
	MODEL.DIST_TRAIN True SOLVER.BASE_LR $lr SOLVER.IMS_PER_BATCH $bs \
	OUTPUT_DIR "${root}_bs_${bs}_lr_${lr}" \
	MODEL.USE_FUSION True MODEL.FUSION_METHOD 'cat'

    done
done


#CUDA_LAUNCH_BLOCKING=1 python train.py --config_file configs/Flare/dc_former.yml 
# test
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
