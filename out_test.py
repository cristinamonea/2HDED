import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import matplotlib.image
import matplotlib.pyplot as plt
from skimage import measure

from AIFmodels import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel


def main():
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model.load_state_dict(torch.load('Hyper_parameters_Checkpoints_25_Nov/checkpoint_15.model'))

    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, 0.25)


def test(path, model, thre):
    model.eval()
    j=0

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0,'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    avg_psnr = 0.0
    avg_ssim = 0.0
    
    # We load the gt and pred images
    gt_list = []
    pred_list = []
    
    for i, depth in enumerate(gt_list):
        output = pred_list[i]

        depth_edge = edge_detection(depth)
        output_edge = edge_detection(output)
        
        matplotlib.image.imsave('data/1.OurDecoder/GT_depth/GT'+str(j)+'.png', depth.view(depth.size(2),depth.size(3)).data.cpu().numpy())
        matplotlib.image.imsave('data/1.OurDecoder/pred_D/pred'+str(j)+'.png', output.view(output.size(2),output.size(3)).data.cpu().numpy())
        

        save_pred = output_aif.view(output_aif.size(1), output_aif.size(2),output_aif.size(3)).data.cpu().numpy()
        save_pred = np.transpose(save_pred, [1, 2, 0])
        normalization_values = {'mean': np.array([0.485, 0.456, 0.406]),
                                'std': np.array([0.229, 0.224, 0.225])}
        save_pred = save_pred * normalization_values['std'] + normalization_values['mean']
        save_pred = np.clip(save_pred, a_min=0.0, a_max=1.0)
        
        save_gt = aif_img.view(aif_img.size(1), aif_img.size(2),aif_img.size(3)).data.cpu().numpy()
        save_gt = np.transpose(save_gt, [1, 2, 0])
        save_gt = save_gt * normalization_values['std'] + normalization_values['mean']
        save_gt = np.clip(save_gt, a_min=0.0, a_max=1.0)
        
        matplotlib.image.imsave('data/1.OurDecoder/Aif_ET/aif_'+str(j)+'.png', save_pred)
        matplotlib.image.imsave('data/1.OurDecoder/Aif_GT/GT_aif_'+str(j)+'.png', save_gt)
        #kkkk
        j=j+1
        
        # PSNR ERROR CALCULATION
        img_psnr = measure.compare_psnr(np.squeeze(save_gt), np.squeeze(save_pred))
        img_ssim = measure.compare_ssim(np.squeeze(save_gt), np.squeeze(save_pred), multichannel=True, data_range=1.0)
        print('[PSNR and SSIM]: ', img_psnr, ' ', img_ssim)
        avg_psnr = avg_psnr + img_psnr
        avg_ssim = avg_ssim + img_ssim
        # PSNR ERROR CALCULATION

        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

        edge1_valid = (depth_edge > thre)
        edge2_valid = (output_edge > thre)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / (depth.size(2)*depth.size(3))

        nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
        P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
        R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

        F = (2 * P * R) / (P + R)

        Ae += A
        Pe += P
        Re += R
        Fe += F

    Av = Ae / totalNumber
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print('PV', Pv)
    print('RV', Rv)
    print('FV', Fv)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)
    
    avg_psnr = avg_psnr / totalNumber
    avg_ssim = avg_ssim / totalNumber
    print('averge PSNR: {:.3f}'.format(avg_psnr))
    print('averge SSIM: {:.3f}'.format(avg_ssim))

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
