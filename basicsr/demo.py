# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import time

# from basicsr.data import create_dataloader, create_dataset
import os
from tqdm import tqdm
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
import shutil

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    
    imgf_path = opt['img_path'].get('input_img')
    outputf_path = opt['img_path'].get('output_img')
    
    if os.path.exists(outputf_path):
        shutil.rmtree(outputf_path)
    os.makedirs(outputf_path)
    list_file = sorted(os.listdir(imgf_path))
    model = create_model(opt)
    
    for each in tqdm(list_file):
        img_path = os.path.join(imgf_path, each)
        output_path = os.path.join(outputf_path, each)

        # if "val" not in each: continue
        # breakpoint()
        # if 'val' not in each: continue


        ## 1. read image
        file_client = FileClient('disk')

        img_bytes = file_client.get(img_path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        img = img2tensor(img, bgr2rgb=True, float32=True)


        # breakpoint()
        ## 2. run inference
        opt['dist'] = False
        # model = create_model(opt)

        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            breakpoint()
            model.grids()
        start_time = time.time()
        model.test()

        end_time = time.time()

        time_taken = end_time - start_time
        print(f'Time taken for inference: {time_taken} seconds')

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, output_path)

        # print(f'inference {img_path} .. finished. saved to {output_path}')
        

if __name__ == '__main__':
    main()

