import os, argparse

import numpy as np

import torch
from torchvision import transforms

# import models.ResNet_staircase as ResNet_staircase
import models.my_model as my_model
from PIL import Image


def parse_args():
    """
    Parse input arguments.
        CUDA_VISIBLE_DEVICES=0 python -u test_staircase.py \
        --test_image_name image_name \
        --model_path model_file \
        --trained_database Koniq10k \
        --test_method five \
        --output_name output.txt

    python test_staircase.py --test_image_name "E:\IQA-github\StairIQA\images\12949713.jpg"
                             --model_path "E:\IQA-github\StairIQA\ResNet_staircase_50-EXP1-Koniq10k.pkl"
                             --trained_database Koniq10k
                             --test_method five
                             --output_name "E:\IQA-github\StairIQA\output.txt"
    """
    parser = argparse.ArgumentParser(description="Authentic Image Quality Assessment")
    parser.add_argument('--model_path', help='Path of model snapshot.',
                        default=r'E:\IQA-github\StairIQA\train-ind-LIVE_challenge-my_model-exp_id-0.pkl', type=str)
    # parser.add_argument('--test_image_name', type=str, default=r'E:\IQA-github\StairIQA\images\203950866.jpg')
    parser.add_argument('--test_image_name', type=str, default=r'E:\IQA-github\StairIQA\images\LIVE_challenge\266.bmp')
    parser.add_argument('--trained_database', default='LIVE_challenge', type=str)
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image (default: one)')
    parser.add_argument('--output_name', type=str, default=r'E:\IQA-github\StairIQA\output.txt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    test_image_name = args.test_image_name
    model_path = args.model_path
    trained_database = args.trained_database

    output_name = args.output_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the network
    model = my_model.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if trained_database == 'FLIVE':
        if args.test_method == 'one':
            transformations_test = transforms.Compose([transforms.Resize(340), transforms.CenterCrop(320), \
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test = transforms.Compose([transforms.Resize(340), transforms.FiveCrop(320), \
                                                       (lambda crops: torch.stack(
                                                           [transforms.ToTensor()(crop) for crop in crops])), \
                                                       (lambda crops: torch.stack(
                                                           [transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                                                 std=[0.229, 0.224, 0.225])(crop) for
                                                            crop in crops]))])
    elif trained_database == 'FLIVE_patch':
        if args.test_method == 'one':
            transformations_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), \
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test = transforms.Compose([transforms.Resize(256), transforms.FiveCrop(224), \
                                                       (lambda crops: torch.stack(
                                                           [transforms.ToTensor()(crop) for crop in crops])), \
                                                       (lambda crops: torch.stack(
                                                           [transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                                                 std=[0.229, 0.224, 0.225])(crop) for
                                                            crop in crops]))])
    else:
        if args.test_method == 'one':
            transformations_test = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(320), \
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test = transforms.Compose([transforms.Resize(384), transforms.FiveCrop(320), \
                                                       (lambda crops: torch.stack(
                                                           [transforms.ToTensor()(crop) for crop in crops])), \
                                                       (lambda crops: torch.stack([transforms.Normalize(
                                                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop)
                                                                                   for crop in crops]))])

    test_image = Image.open(test_image_name)
    test_image = test_image.convert('RGB')
    test_image = transformations_test(test_image)
    test_image = test_image.unsqueeze(0)

    with torch.no_grad():
        if args.test_method == 'one':
            test_image = test_image.to(device)
            if trained_database == 'FLIVE':
                outputs, _, _, _, _, _ = model(test_image)
            elif trained_database == 'FLIVE_patch':
                _, outputs, _, _, _, _ = model(test_image)
            elif trained_database == 'LIVE_challenge':
                _, _, outputs, _, _, _ = model(test_image)
            elif trained_database == 'Koniq10k':
                _, _, _, outputs, _, _ = model(test_image)
            elif trained_database == 'SPAQ':
                _, _, _, _, outputs, _ = model(test_image)
            elif trained_database == 'BID':
                _, _, _, _, _, outputs = model(test_image)
            test_scores = outputs.item()
            print(test_image_name)
            print(test_scores)


        elif args.test_method == 'five':
            bs, ncrops, c, h, w = test_image.size()
            itest_imagemage = test_image.to(device)
            if trained_database == 'FLIVE':
                outputs, _, _, _, _, _ = model(test_image.view(-1, c, h, w))
            elif trained_database == 'FLIVE_patch':
                _, outputs, _, _, _, _ = model(test_image.view(-1, c, h, w))
            elif trained_database == 'LIVE_challenge':
                _, _, outputs, _, _, _ = model(test_image.view(-1, c, h, w))
            elif trained_database == 'Koniq10k':
                _, _, _, outputs, _, _ = model(test_image.view(-1, c, h, w))
            elif trained_database == 'SPAQ':
                _, _, _, _, outputs, _ = model(test_image.view(-1, c, h, w))
            elif trained_database == 'BID':
                _, _, _, _, _, outputs = model(test_image.view(-1, c, h, w))
            test_scores = outputs.view(bs, ncrops, -1).mean(1).item()
            print(test_image_name)
            print(test_scores)

    # if not os.path.exists(output_name):
    #     os.system(r"touch {}".format(output_name))
    #
    # f = open(output_name, 'w')
    # f.write(test_image_name)
    # f.write(',')
    # f.write(str(test_scores))
    # f.write('\n')
    #
    # f.close()

    # 如果文件不存在则创建文件
    if not os.path.exists(output_name):
        os.system(r"touch {}".format(output_name))

    # 使用 'a' 模式打开文件，表示追加内容
    with open(output_name, 'a') as f:
        f.write(test_image_name)
        f.write(',')
        f.write(str(test_scores))
        f.write('\n')

    f.close()
