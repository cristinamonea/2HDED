import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import matplotlib.image
import matplotlib.pyplot as plt
from skimage import measure
# import skimage.measure as measure
#
# from AIFmodels import modules, net, resnet, densenet, senet
# import loaddata
# from Errors import util
import numpy as np
# import sobel
import os
from PIL import Image
### Debluring Metrics
# from sewar.full_ref import uqi
# from sewar.full_ref import vifp
# from sewar.full_ref import psnr
# import lpips
# from sewar.full_ref import ssim

# from piq import psnr
# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# d = loss_fn_alex(img0, img1)






# def main():
#     model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
#     # model = torch.nn.DataParallel(model).cuda()
#     model = model.cuda()
#     model.load_state_dict(torch.load('Hyper_parameters_Checkpoints_25_Nov/checkpoint_15.model'))
#
#     test_loader = loaddata.getTestingData(1)
#     test(test_loader, model, 0.25)
#

def test(path, model, thre):
    model.eval()
    j = 0

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    # We load the gt and pred images
    gt_list = 'D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/target/depth'
    pred_list = 'D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/output/depth'
    

    gt_list = [file for file in os.listdir(gt_list) if file.endswith('.png')]
    pred_list = [file for file in os.listdir(pred_list) if file.endswith('.png')]

    target = []
    for i in gt_list:
        img_temp = np.array(Image.open("D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/target/depth/%s" %i), dtype=float)
        target.append(img_temp)

    prediction = []
    for i in pred_list:
        img_temp = np.array(Image.open("D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/output/depth/%s" %i), dtype=float)
        prediction.append(img_temp)

    # Errors = util.evaluateError(output=prediction, target=target)
    
   

  
    # plt.figure()
    # plt.imshow(aa[0])
    # plt.show()
    # plt.figure()
    # plt.imshow(bb[0])
    # plt.show()
    # img1 = target[0]
    # img2 = prediction[0]
    # img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
    # img2 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))
    # L = Error(torch.tensor(img1), torch.tensor(img2))
    # print(L)
    def nValid(x):
        return torch.sum(torch.eq(x, x).float())

    def nNanElement(x):
        return torch.sum(torch.ne(x, x).float())
    
    def getNanMask(x):
        return torch.ne(x, x)

    def setNanToZero(input, target):
        nanMask = getNanMask(target)
        # nValidElement = nValid(target)
        nValidElement = target.size(0)*target.size(1)
        
        _input = input.clone()
        _target = target.clone()
    
        _input[nanMask] = 0
        _target[nanMask] = 0
    
        return _input, _target, nanMask, nValidElement
    
    
    ########Calculating the errosrs and saving images from np arry for Depth
    #G:\1.Final_9_OCTOBER_Normalizedv1v2\ALL_IN_FOCUS
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    
    gt_list = 'G:/1.Final_9_OCTOBER_Normalizedv1v2/ALL_IN_FOCUS'
    pred_list = 'G:/1.Final_9_OCTOBER_Normalizedv1v2/OUT_OF_FOCUS'
    

    gt_list = [file for file in os.listdir(gt_list) if file.endswith('.JPG')]
    pred_list = [file for file in os.listdir(pred_list) if file.endswith('.JPG')]

    target = []
    for i in gt_list:
        img_temp = np.load("G:/1.Final_9_OCTOBER_Normalizedv1v2/ALL_IN_FOCUS/%s" %i)
        img_temp = np.moveaxis(img_temp, source=0, destination=2)
        target_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
       
        # target.append(img_temp)
        # ####Save as PNG
        # target_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/gt_depth/%s.png' %i, target_temp)
   
    prediction = []
    for i in pred_list:
        img_temp = np.load("G:/1.Final_9_OCTOBER_Normalizedv1v2/OUT_OF_FOCUS/%s" %i)
        img_temp = np.moveaxis(img_temp, source=0, destination=2)
        prediction_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        
        # prediction.append(img_temp)
        # ####Save as PNG
        # prediction_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/pred_depth/%s.png' %i, prediction_temp)
    

    MSE = []
    MAE = []
    REL = []
    for i, depth in enumerate(gt_list):
        # output = pred_list[i]
        target_temp = target[i]
        target_temp = target_temp
        prediction_temp = prediction[i]
        ##################
        # rmse_log = (np.log(target_temp) - np.log(prediction_temp)) ** 2
        
    
        # abs_rel = np.abs(target_temp - prediction_temp) / target_temp
    
        # sq_rel =((target_temp - prediction_temp) ** 2) / target_temp
        
        
        # ###########
        # thresh = np.maximum((target_temp / prediction_temp), (prediction_temp / target_temp))
        
        # a1 = (thresh < 1.25)
        # a2 = (thresh < 1.25 ** 2)
        # a3 = (thresh < 1.25 ** 3)
        # print(a1)
        # print(a2)
        # print(a3)

        ###########
        #target_temp = torch.tensor(target_temp)
        #prediction_temp = torch.tensor(prediction_temp)
        target_temp = torch.tensor((target_temp - np.min(target_temp)) / (np.max(target_temp) - np.min(target_temp)))
        prediction_temp = torch.tensor((prediction_temp - np.min(prediction_temp)) / (np.max(prediction_temp) - np.min(prediction_temp)))
               
        #MSE
        MSE_error = torch.nn.MSELoss()
        MSE_temp = MSE_error(target_temp, prediction_temp)
        MSE.append(MSE_temp.detach().numpy())
        
        #MAE
        MAE_error = torch.nn.L1Loss()        
        MAE_temp = MAE_error(target_temp, prediction_temp)
        MAE.append(MAE_temp.detach().numpy())
       
        
       
        
        # acc = torch.maximum(prediction_temp/target_temp, target_temp/prediction_temp)
        # delta1 = (acc < 1.25).float()
         
        # delta2 = (acc < 1.25**2).float()
        # # print('Delta2: {:.6f}'.format(delta2))
        # delta3 = (acc < 1.25**3).float()
        # #REL
        # numElement = target_temp.size(0)*target_temp.size(1)
        # REL_temp = (torch.sum(torch.abs(target_temp - prediction_temp)/torch.abs(target_temp)))/numElement
        # REL.append(REL_temp.detach().numpy())
        
        # _output, _target, nanMask, nValidElement = setNanToZero(torch.tensor(prediction_temp), torch.tensor(target_temp))
        # diffMatrix = torch.abs(_output - _target)
        # realMatrix = torch.div(diffMatrix, _target)
        # realMatrix[nanMask] = 0.00000000001
        # REL_temp = torch.sum(realMatrix) / nValidElement         
        # REL.append(REL_temp.detach().numpy())
        
        ####Log10
        # Log10 = torch.mean(torch.abs(torch.log10(prediction_temp)-torch.log10(target_temp)))
        # print("Log10:", Log10) 
        
        
        
        ################# RMSE 2 Good ######
        rmse = torch.sqrt(torch.mean((prediction_temp - target_temp)**2))
        # print('Root Mean Squared Error: {:.6f}'.format(rmse))
        # log = torch.sub(torch.log10(prediction_temp), torch.log10(target_temp))
        # ablog = torch.abs(log)
        
        
        # log10 = torch.mean(torch.abs(torch.log10(prediction_temp) - torch.log10(target_temp)))
        # print('Mean Log10 Error: {:.6f}'.format(log10))
        
        ####Delta 1 2 3 #########
        # yOverZ = torch.div(prediction_temp, target_temp)
        # zOverY = torch.div(target_temp, prediction_temp)
        # maxRatio = torch.maximum(yOverZ, zOverY)
        # delta1 = torch.le(maxRatio , 1.25).float()
        # delta2 = torch.le(maxRatio , 1.25**2).float()
        # delta3 = torch.le(maxRatio , 1.25**3).float()
        # errors['DELTA1'] = torch.sum(
        # torch.le(maxRatio, 1.25).float())
        
        # acc = torch.maximum(prediction_temp/target_temp, target_temp/prediction_temp)
        # delta1 = (acc < 1.25).float()
         
        # delta2 = (acc < 1.25**2).float()
        # # print('Delta2: {:.6f}'.format(delta2))
        # delta3 = (acc < 1.25**3).float()
        # print('Delta3: {:.6f}'.format(delta3))
        # rel_error = torch.abs(prediction_temp - target_temp)/target_temp
        # print('\nMean Absolute Relative Error: {:.6f}', rel_error)
 


        
        
    #####
    rmse_log = np.sqrt(rmse_log.mean())
    print("RMSE_LOG:", rmse_log)
    abs_rel_avg = np.mean(abs_rel)
    print("ABS_REL:", abs_rel_avg)
    sq_rel_avg = np.mean(sq_rel)
    print("SQ_REL:", sq_rel_avg)
    
    #####
    
    avg_MSE_error = np.average(MSE)
    RMSE = np.sqrt(avg_MSE_error)
    print("RMSE:", RMSE)

    avg_MAE_error = np.average(MAE)
    print("MAE:", avg_MAE_error)
    
    # avg_REL_error = np.average(rel)
    # print("REL:", rel)
    # avg_log10=torch.mean(Log10)
    # print("Log10:", Log10)
    rmse_average = np.average(rmse)
    print("RMSE2:", rmse_average)
    
    # avg_log10=torch.mean(ablog)
    # print('Mean Log10 Error: ',avg_log10)
    
    delta1_avg = np.mean(a1)
    print('Delta1:', delta1_avg )
    delta2_avg =np.mean(a2) 
    print('Delta2:', delta2_avg )
    delta3_avg = np.mean(a3)
    print('Delta2:', delta3_avg )
    
    
    
    import cv2
    
    
    # PSNR ERROR CALCULATION
     # We load the gt and pred images
    avg_psnr = 0.0
    avg_ssim = 0.0
    gt_list = 'G:/1.Final_9_OCTOBER_Normalizedv1v2/ALL_IN_FOCUS'
    pred_list = 'G:/1.Final_9_OCTOBER_Normalizedv1v2/OUT_OF_FOCUS'
    

    gt_list = [file for file in os.listdir(gt_list) if file.endswith('.JPG')]
    pred_list = [file for file in os.listdir(pred_list) if file.endswith('.JPG')]

    target = []
    for i in gt_list:
        img_temp = cv2.imread("G:/1.Final_9_OCTOBER_Normalizedv1v2/ALL_IN_FOCUS/%s" %i)
        # img_temp = np.moveaxis(img_temp, source=2, destination=4)
        target.append(img_temp)
        ####Save as PNG
        # target_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/aif_data/%s.png' %i, target_temp)
   
    prediction = []
    for i in pred_list:
        img_temp = cv2.imread("G:/1.Final_9_OCTOBER_Normalizedv1v2/OUT_OF_FOCUS/%s" %i)
        # img_temp = np.moveaxis(img_temp, source=2, destination=4)
        prediction.append(img_temp)
        ####Save as PNG
        # prediction_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/aif_pred/%s.png' %i, prediction_temp)
       
        
    for i in range(len(target)):
       
       
        img_psnr = measure.compare_psnr(np.squeeze(target[i]), np.squeeze(prediction[i]))
        img_ssim = measure.compare_ssim(np.squeeze(target[i]), np.squeeze(prediction[i]), multichannel=True, data_range=1.0)
        print('[PSNR]: ', img_psnr)
        avg_psnr = avg_psnr + img_psnr
        avg_ssim = avg_ssim + img_ssim
        ##### PSNR = psnr(np.squeeze(target[i]),np.squeeze(prediction[i]),MAX=None)
        # VIF =  vifp(np.squeeze(target[i]),np.squeeze(prediction[i]))
        # UQI =  uqi(np.squeeze(target[i]),np.squeeze(prediction[i]))
        ### print('[VIF and QUI]: ', VIF, ' ', UQI)
       ##  # best forward scores
       # # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
       # # a = torch.tensor(target[i])
       # # b = torch.tensor(prediction[i])
       # # sewar.full_ref =ssim(a, b, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid')
        
        
        loss_fn_alex = lpips.LPIPS(net='alex')
        # print(target.dtype)
        # print(target.shape)
        # a = torch.squeeze(target)
        # a = torch.permute(a, (2,0,1))
        # b = torch.squeeze(prediction)
        # b = torch.permute(b, (2,0,1))
        # # print(b.dtype)
        lpp = loss_fn_alex(a[i],b[i])
        # print("lp",lpp)
        # print(b.shape)
        # target = torch.from_numpy(target[i])
        # prediction = torch.from_numpy(prediction[i])
        # target_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # prediction_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
    
    
    avg_PSNR_error = np.average(img_psnr)
    print("PSNR_Avg:", avg_PSNR_error)
    avg_SSIM_error = np.average(img_ssim)
    print("SSIM_Avg:", avg_SSIM_error)
    avg_VIF_error = np.average(VIF)
    print("VIF_Avg:", avg_VIF_error)
    avg_UQI_error = np.average(UQI)
    print("UQI_Avg:", avg_UQI_error)
    # avg_lpp_error = torch.mean(lpp)
    # print("LPIPS_Average:", avg_lpp_error)
    
    
    
    
    #     batchSize = depth.size(0)
    #     totalNumber = totalNumber + batchSize
    #     errors = util.evaluateError(output, depth)
    #     errorSum = util.addErrors(errorSum, errors, batchSize)
    #     averageError = util.averageErrors(errorSum, totalNumber)
    #
    #     edge1_valid = (depth_edge > thre)
    #     edge2_valid = (output_edge > thre)
    #
    #     nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
    #     A = nvalid / (depth.size(2) * depth.size(3))
    #
    #     nvalid2 = np.sum(((edge1_valid + edge2_valid) == 2).float().data.cpu().numpy())
    #     P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
    #     R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))
    #
    #     F = (2 * P * R) / (P + R)
    #
    #     Ae += A
    #     Pe += P
    #     Re += R
    #     Fe += F
    #
    # Av = Ae / totalNumber
    # Pv = Pe / totalNumber
    # Rv = Re / totalNumber
    # Fv = Fe / totalNumber
    # print('PV', Pv)
    # print('RV', Rv)
    # print('FV', Fv)
    #
    # averageError['RMSE'] = np.sqrt(averageError['MSE'])
    # print(averageError)
    #
    # avg_psnr = avg_psnr / totalNumber
    # avg_ssim = avg_ssim / totalNumber
    # print('averge PSNR: {:.3f}'.format(avg_psnr))
    # print('averge SSIM: {:.3f}'.format(avg_ssim))


#######Loading and testing depths
GT_list = 'D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/output/depth'
Pred_list = 'D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/results/multitask/nyu_test/latest/target/depth'

a = a.squeeze()
a = np.moveaxis(a, source=0, destination=2)

# b = b.squeeze()
# b = np.moveaxis(b, source=0, destination=2)

aa = (a - np.min(a))/(np.max(a) - np.min(a))
# bb = (b - np.min(b))/(np.max(b) - np.min(b))


plt.figure()
plt.imshow(a)
plt.show()

# plt.figure()
# plt.imshow(bb)
# plt.show()

