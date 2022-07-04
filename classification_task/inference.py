from dataset import ImageFolderWithPaths

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import pdb
import shutil
import cv2

class_id = ['defect', 'normal']
vis_debug = 'debug_classify_results'

if os.path.isdir(vis_debug) is True:
    shutil.rmtree(vis_debug)

if os.path.isdir(vis_debug) is False:
    os.mkdir(vis_debug)
    os.mkdir(os.path.join(vis_debug, class_id[0]))
    os.mkdir(os.path.join(vis_debug, class_id[1]))

def classify_images(model_conv, path_txt_result, dataloaders):

    list_result = {}

    with torch.no_grad():
        for i, (inputs, labels, img_paths) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)

            for index in range(inputs.shape[0]):
                img_path, pred_class = img_paths[index], preds[index].cpu().item()
                class_name = class_id[pred_class]

                ori_img_name = img_path.split('/')[-1].split('__')[0]
                order_chicker = img_path.split('/')[-1].split('__')[1]
                if ori_img_name not in list_result:
                    list_result[ori_img_name] = {}

                list_result[ori_img_name][order_chicker] = class_name
                img_name = img_path.split('.')[0]
                image_result_name = img_name.split('/')[-1]+"___"+class_name+'.jpg'

                # for debug only
                img = cv2.imread(img_path)

                # write to path for debugging result
                path_write = os.path.join(vis_debug, class_name, image_result_name)
                cv2.imwrite(path_write, img)

    # sort the result 
    
    with open(path_txt_result, 'w') as fp:
        for filename in list_result:

            result_predict_str = ''
            filename_with_tag = filename+'.jpg'

            result_pred_instance = []
            for instance in sorted(list_result[filename].keys()):
                result_pred_instance.append(list_result[filename][instance])

            result_predict_str = ','.join(result_pred_instance)

            fp.write(filename_with_tag+' '+result_predict_str)
            fp.write('\n')



if __name__ == '__main__':
    final_test_mask_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/final_test_mask'
    crop_test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/classification/new_test'
    test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/dataset/Dataset/classification_data/test'
    
    prediction_path = '/home/nttung/BB/Instance_Semantic_Segmentation/classification/classification.txt'
    sub_crop_test_path = '/home/nttung/BB/Instance_Semantic_Segmentation/classification/'
    ### First define model again and load checkpoint
    ### Build model

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv = model_conv.to(device)

    model_conv.load_state_dict(torch.load('/home/nttung/BB/Instance_Semantic_Segmentation/classification/checkpoints/14.pt'))
    model_conv.eval()

    ### Build test dataloader
    data_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    img_data_test = ImageFolderWithPaths(sub_crop_test_path, data_transform)
    img_dataloader_test = torch.utils.data.DataLoader(img_data_test, batch_size=4,
                                             shuffle=False, num_workers=4)
    

    classify_images(model_conv, prediction_path, img_dataloader_test)