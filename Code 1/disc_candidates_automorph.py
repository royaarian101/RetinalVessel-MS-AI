import os, json, sys
import os.path as osp
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torchvision
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model
from skimage import measure
import pandas as pd
from skimage.morphology import remove_small_objects
# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--config_file', type=str, default=None,
                    help='experiments/name_of_config_file, overrides everything')
# in case no config file is passed
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--results_path', type=str, default='results', help='path to save predictions (defaults to results')

        
def prediction_eval(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8, test_loader):
    
    n_val = len(test_loader)
    
    seg_results_small_path = '../OD_candidates/'
   
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    


        
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            imgs = batch['image']
            img_name = batch['name']
            ori_width=batch['original_sz'][0]
            ori_height=batch['original_sz'][1]
            mask_pred_tensor_small_all = 0
            
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():

                _,mask_pred = model_1(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_1 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_1.type(torch.FloatTensor)
                
                
                _,mask_pred= model_2(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_2 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_2.type(torch.FloatTensor)
                

                _,mask_pred = model_3(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_3 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_3.type(torch.FloatTensor)                
                

                _,mask_pred = model_4(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_4 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_4.type(torch.FloatTensor)    
                

                _,mask_pred = model_5(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_5 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_5.type(torch.FloatTensor)    
                

                _,mask_pred = model_6(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_6 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_6.type(torch.FloatTensor)   
                

                _,mask_pred = model_7(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_7 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_7.type(torch.FloatTensor)   
                

                _,mask_pred = model_8(imgs)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small_8 = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small_all+=mask_pred_tensor_small_8.type(torch.FloatTensor)   
                
                mask_pred_tensor_small_all = (mask_pred_tensor_small_all/8).to(device=device)
                
                
                uncertainty_map = torch.sqrt((torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_1)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_2)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_3)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_4)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_5)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_6)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_7)+torch.square(mask_pred_tensor_small_all-mask_pred_tensor_small_8))/8)
            
                _,prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                prediction_decode=prediction_decode.type(torch.FloatTensor)
                
                n_img = prediction_decode.shape[0]
                
                if len(prediction_decode.size())==3:
                    torch.unsqueeze(prediction_decode,0)
                if len(uncertainty_map.size())==3:
                    torch.unsqueeze(uncertainty_map,0)
                
                for i in range(n_img):
                    
                   # save_image(mask_pred_tensor_small_all[i,0,..]*255, seg_results_small_path+img_name[i]+'_channel0.png')
                   # save_image(mask_pred_tensor_small_all[i,1,...]*255, seg_results_small_path+img_name[i]+'.png')#first channel
                    save_image(mask_pred_tensor_small_all[i,2,...]*255, seg_results_small_path+img_name[i]+'.png')#second channel
                    
                   
                    
                    img_r = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_g = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    img_b = np.zeros((prediction_decode[i,...].shape[0],prediction_decode[i,...].shape[1]))
                    
                    
                    img_r[prediction_decode[i,...]==1]=255
                    img_b[prediction_decode[i,...]==2]=255

                    img_b = remove_small_objects(img_b>0, 50)
                    img_r = remove_small_objects(img_r>0, 100)

                    img_ = np.concatenate((img_b[...,np.newaxis], img_g[...,np.newaxis], img_r[...,np.newaxis]), axis=2)
                    
                    
                
                
                pbar.update(imgs.shape[0])
                
                


if __name__ == '__main__':

    args = parser.parse_args()
    results_path = args.results_path
    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        device = torch.device("cuda")
    else:  # cpu
        device = torch.device(args.device)


    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)
    experiment_path = args.experiment_path  # this should exist in a config file
    model_name = args.model_name

    if experiment_path is None: raise Exception('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    data_path = '../preprocessed_OD/'

    csv_path = 'test_all.csv'
    test_loader = get_test_dataset(data_path, csv_path=csv_path, tg_size=tg_size)
    
    model_1 = get_arch(model_name, n_classes=3).to(device)
    model_2 = get_arch(model_name, n_classes=3).to(device)
    model_3 = get_arch(model_name, n_classes=3).to(device)
    model_4 = get_arch(model_name, n_classes=3).to(device)
    model_5 = get_arch(model_name, n_classes=3).to(device)
    model_6 = get_arch(model_name, n_classes=3).to(device)
    model_7 = get_arch(model_name, n_classes=3).to(device)
    model_8 = get_arch(model_name, n_classes=3).to(device)
    
    experiment_path_1 = './experiments/wnet_All_three_1024_disc_cup/28/'
    experiment_path_2 = './experiments/wnet_All_three_1024_disc_cup/30/'
    experiment_path_3 = './experiments/wnet_All_three_1024_disc_cup/32/'
    experiment_path_4 = './experiments/wnet_All_three_1024_disc_cup/34/'
    experiment_path_5 = './experiments/wnet_All_three_1024_disc_cup/36/'
    experiment_path_6 = './experiments/wnet_All_three_1024_disc_cup/38/'
    experiment_path_7 = './experiments/wnet_All_three_1024_disc_cup/40/'
    experiment_path_8 = './experiments/wnet_All_three_1024_disc_cup/42/'


    model_1, stats = load_model(model_1, experiment_path_1, device)
    model_1.eval()

    model_2, stats = load_model(model_2, experiment_path_2, device)
    model_2.eval()
    
    model_3, stats = load_model(model_3, experiment_path_3, device)
    model_3.eval()
    
    model_4, stats = load_model(model_4, experiment_path_4, device)
    model_4.eval()
    
    model_5, stats = load_model(model_5, experiment_path_5, device)
    model_5.eval()
    
    model_6, stats = load_model(model_6, experiment_path_6, device)
    model_6.eval()
    
    model_7, stats = load_model(model_7, experiment_path_7, device)
    model_7.eval()
    
    model_8, stats = load_model(model_8, experiment_path_8, device)
    model_8.eval()


    prediction_eval(model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8, test_loader)
    

    