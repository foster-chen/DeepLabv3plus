python main.py --model deeplabv3plus_resnet101\
       	--dataset cityscapes\
       	--lr 0.01\
       	--crop_size 768\
       	--batch_size 16\
       	--output_stride 16\
       	--data_root /opt/cloudroot/datasets/cityscapes\
       	--gpu_id 0,1\
       	--run_name None_FT_Cityscapes_0.01\
       	--wandb
