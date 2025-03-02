import os
import torch
import argparse
import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from densematching.utils_flow.pixel_wise_mapping import remap_using_flow_fields
from densematching.model_selection import select_model
from densematching.utils_flow.util_optical_flow import flow_to_image
from densematching.utils_flow.visualization_utils import overlay_semantic_mask
# from validation.test_parser import define_model_parser


def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)

    return im1, im2


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def DenseMatching(opt, train_data):
    # blur, sharps = train_data
    # breakpoint()
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            opt['model'], opt['pre_trained_model'],opt['PDCNet'], opt['optim_iter'], opt['local_optim_iter'],
            path_to_pre_trained_models="densematching/pre_trained_models")
        # breakpoint()
        inp = {"lq":    [],
                "gt":   [],
                "conf": [],}
        breakpoint()
        for inputs in zip(train_data[0], train_data[1]):
            breakpoint()
            blur, sharps = inputs
            
            mid = 9
            final = None
            ts = []
            # ms = []
            epsilon = 1e-8
            for idx, query in tqdm(enumerate(sharps)):
                # if each == mid: continue
                # try:
                #     if each > 9:
                #         query_path = f"images/datasets/REDS/original/train/train_sharp/014/000000{each}.png"
                #     else: query_path = f"images/datasets/REDS/original/train/train_sharp/014/0000000{each}.png"
                #     query_image = cv2.imread(query_path, 1)[:, :, ::- 1]
                #     reference_image = cv2.imread(args.path_reference_image, 1)[:, :, ::- 1]
                # except:
                #     raise ValueError('It seems that the path for the images you provided does not work ! ')
                if final is not None:
                    blur = final 
                conf1d, warped_query_image = test_model_on_image_pair(opt, network, estimate_uncertainty, query, blur, idx, mid)
                final = blur * (1 - conf3d) + warped_query_image * conf1d[...,np.newaxis] 
                ts.append(final)
                # ms.append(conf1d)

                # images = np.array(ts)      # Shape: (n, height, width, channels)
                # conf_maps = np.array(ms)
                # weighted_sum = np.sum(images * conf_maps[..., np.newaxis], axis=0)
                # conf_sum = np.sum(conf_maps, axis=0) + epsilon

                # avg_img = weighted_sum / conf_sum[..., np.newaxis]
                # cv2.imwrite("avg.png",avg_img.astype(int))
                # breakpoint()
            # print(11111)
            # print(mid)
            # breakpoint()
            images = np.array(ts)      # Shape: (n, height, width, channels)
            conf_maps = np.array(ms)
            # weighted_sum = np.sum(images * conf_maps[..., np.newaxis], axis=0)
            # conf_sum = np.sum(conf_maps, axis=0) + epsilon

            # avg_img = weighted_sum / conf_sum[..., np.newaxis]
            # image = cv2.convertScaleAbs(avg_img, alpha=(255.0/ np.max(avg_img)))  
            # temp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
            # cv2.imwrite("avg.png",temp)

            confi_map = np.all(final == [0, 0, 0], axis=-1)  
            confi_map = confi_map.astype(int)
            total_conf = (1 - confi_map) * conf_maps[0]
            avg = final * total_conf[..., np.newaxis] # RGB
            avg = avg.astype(np.uint8)
            inp['lq'].append(torch.from_numpy(blur).permute(2, 0, 1))
            inp['gt'].append(torch.from_numpy(avg).permute(2, 0, 1))
            inp['conf'].append(torch.from_numpy(total_conf).permute(2, 0, 1))

        inp =  {k: torch.tensor(v) for k, v in inp.items()}

        return inp
        # breakpoint()
        # u = 1 
        
            # final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)     

def test_model_on_image_pair(args, network, estimate_uncertainty, query_image, reference_image, idx, mid):
    # save original ref image shape
    ref_image_shape = reference_image.shape[:2]

    # pad both images to the same size, to be processed by network
    query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
    # convert numpy to torch tensor and put it in right format
    query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
    reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

    # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
    # specific pre-processing (/255 and rescaling) are done within the function.

    # pass both images to the network, it will pre-process the images and ouput the estimated flow
    # in dimension 1x2xHxW
    if estimate_uncertainty:
        # if args.flipping_condition:
        #     raise NotImplementedError('No flipping condition with PDC-Net for now')

        estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                            reference_image_,
                                                                                            mode='channel_first')
        confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
        confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]
    else:
        pass
        # if args.flipping_condition and 'GLUNet' in args.model:
        #     estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
        #                                                                     mode='channel_first')
        # else:
        #     estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
    estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
    estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
    # removes the padding

    warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                    estimated_flow_numpy[:, :, 1]).astype(np.uint8)

    # save images
    # if args.save_ind_images:
    #     imageio.imwrite(os.path.join(args.save_dir, 'query.png'), query_image)
    #     imageio.imwrite(os.path.join(args.save_dir, 'reference.png'), reference_image)
        # imageio.imwrite(os.path.join(args.save_dir, 'warped_query_{}_{}.png'.format(args.model, args.pre_trained_model)),
        #                 warped_query_image)
    confident_mask = (confidence_map > 0.53).astype(np.uint8)
    # if estimate_uncertainty:
    #     color = [255, 102, 51]
    #     fig, axis = plt.subplots(1, 5, figsize=(30, 30))

    #     confident_mask = (confidence_map > 0.53).astype(np.uint8)
    #     # confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)

    #     # breakpoint()
    #     axis[2].imshow(confident_warped)
    #     axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
    #                         .format(opt['model'], opt['pre_trained_model']))
    #     axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
    #     axis[4].set_title('Confident regions')
    # else:
    #     fig, axis = plt.subplots(1, 4, figsize=(30, 30))
    #     axis[2].imshow(warped_query_image)
    #     axis[2].set_title(
    #         'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
    # axis[0].imshow(query_image)
    # axis[0].set_title('Query image')
    # axis[1].imshow(reference_image)
    # axis[1].set_title('Reference image')

    # axis[3].imshow(flow_to_image(estimated_flow_numpy))
    # axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
    # if not os.path.isdir(os.path.join(args.save_dir, str(mid) + "_" + str(args.ranging))):
    #     os.makedirs(os.path.join(args.save_dir, str(mid) + "_" + str(args.ranging)))
    # fig.savefig(
    #     os.path.join(args.save_dir, str(mid) + "_" + str(args.ranging), 'Warped_image_{}_{}_{}.png'.format(args.model, args.pre_trained_model, str(idx))),
    #     bbox_inches='tight')
    # plt.close(fig)
    # print('Saved image!')

    # conf3d = np.stack((confident_mask,) * 3, axis=-1)        
    return confident_mask, warped_query_image


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description='Test models on a pair of images')
    # define_model_parser(parser)  # model parameters
    # parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model', required=True)
    # parser.add_argument('--path_query_image', type=str,
    #                     help='Path to the source image.', required=True)
    # parser.add_argument('--path_reference_image', type=str,
    #                     help='Path to the target image.', required=True)
    # parser.add_argument('--save_dir', type=str, required=True,
    #                     help='Directory where to save output figure.')
    # parser.add_argument('--save_ind_images', dest='save_ind_images',  default=False, type=boolean_string,
    #                     help='Save individual images? ')
    # parser.add_argument('--ranging', type=int, help='Name of the pre-trained-model', required=True)
    # args = parser.parse_args()

    # torch.cuda.empty_cache()
    # torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    # torch.backends.cudnn.enabled = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    # local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)

    # if not os.path.exists(args.path_query_image):
    #     raise ValueError('The path to the source image you provide does not exist ! ')
    # if not os.path.exists(args.path_reference_image):
    #     raise ValueError('The path to the target image you provide does not exist ! ')

    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)
    
    # test_model(args)

        # final = reference_image * (1 - conf3d) + warped_query_image*conf3d    
        # img = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)   
        


