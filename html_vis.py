import html
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Description of Visualizer')

# Add arguments
parser.add_argument('-n', type=str, help='Name of HTML file.')
parser.add_argument('-blur',  type=str, help='Name of blur path')
parser.add_argument('-sharp',  type=str, help='Name of sharp path')
parser.add_argument('-gt',  type=str, help='Name of gt path')

# Parse the arguments
args = parser.parse_args()

# Access the values of the arguments

web_dir = ""
webpage = html.HTML(web_dir, args.n)

width = 400
# img_size = 448
# image_dir, image_dir_realA = webpage.get_image_dir()

short_path = ntpath.basename(image_path[0])
name = os.path.splitext(short_path)[0]



webpage.add_header(name)
ims, txts, links = [], [], []
ims_dict = {}
for label, im_data in visuals.items():
    if label not in ['fake_B', 'real_A']: continue
    im_dir = image_dir if label == 'fake_B' else image_dir_realA
    # im_data = transforms.functional.crop(im_data, 0, 0, sizeA[1], sizeA[0])
    im = util.tensor2im(im_data)

    # image_name = '%s_%s.png' % (name, label)
    image_name = '%s.png' % (name)
    save_path = os.path.join(im_dir, image_name)
    # if (label == 'fake_B'):
    util.save_image(im, save_path, aspect_ratio=aspect_ratio)
    ims.append(image_name)
    txts.append(label)        
    # breakpoint()
    links.append(im_dir.split("/")[-1] + "/" + image_name)
    if use_wandb:
        ims_dict[label] = wandb.Image(im)
# breakpoint()
webpage.add_images(ims, txts, links, width=width)