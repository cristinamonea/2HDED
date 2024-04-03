# simplified main
from options.extra_args_mtl import MTL_Options as TrainOptions
from dataloader.data_loader import CreateDataLoader

# Load options
opt = TrainOptions(dataroot=r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored",
                   name="simcol3d_480_mirrored_2hded_supervised_L1+L1", 
                   imageSize=[480], 
                   outputSize=[480],
                   resume=False,
                   train=True, 
                   validate=True,
                   test=False,
                   epoch='best', 
                   visualize=False, 
                   test_split='test', 
                   train_split='train', 
                   val_split='val',
                   depth_loss='L1',
                   depth_reg='grad',
                   aif_loss='L1',
                   aif_reg="SSIM",
                   aif_loss_coef=1,
                   depth_reg_coef=0,
                   aif_reg_coef=0,
                   use_dropout=False,
                   use_skips=False,
                   cuda=True, 
                   nEpochs=51,  # Updated to match 'nepochs'
                   batchSize=8,  # Updated to match 'batch_size'
                   init_method='normal', 
                   data_augmentation=["f", "f", "f", "f", "f"], 
                   display=True, 
                   dataset_name="simcol3d",
                   port=8098,  # Updated to match 'port'
                   display_id=100,  # Updated to match 'display_id'
                   display_freq=80, 
                   print_freq=100, 
                   lr = 0.002,
                   checkpoints='checkpoints', 
                   save_samples=True, 
                   save_checkpoint_freq=10,  # Updated to match 'save_ckpt_freq'
                   scale_to_mm = 326.4, #devide by scale_to_mm
                   max_distance = 255,
                   val_freq= 2560, 
                   not_save_val_model=True, 
                   model='depth',  # Updated to match 'model'
                   net_architecture='D3net_multitask',  # Updated to match 'net_architecture'
                   workers=2, 
                   use_resize=False)

# train model
if __name__ == '__main__':
    if opt.train or opt.resume:
        from models.mtl_train import MultiTaskGen as Model
        model = Model()
        model.initialize(opt)
        data_loader, val_loader = CreateDataLoader(opt)
        model.train(data_loader, val_loader=val_loader)
    elif opt.test:
        from models.mtl_test import MTL_Test as Model
        model = Model()
        model.initialize(opt)
        model.test()
