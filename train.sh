# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port 1234 train.py \
--config_file configs/RGBNT201/dc_former.yml \
MODEL.DIST_TRAIN True

#python train.py --config_file configs/RGBNT100/dc_former.yml 
# test
#python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
