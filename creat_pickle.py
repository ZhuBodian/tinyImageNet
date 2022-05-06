import os
import pandas as pd
from PIL import Image
import torch
import utils
import numpy as np
from torchvision import transforms


def creat_pickle(path, transform):
    image_dir = os.path.join(path, 'images')
    csv_modes = ['train', 'test']

    for csv_mode in csv_modes:
        csv_name = csv_mode + '.csv'
        csv_path = os.path.join(path, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found. please run creat_tiny.py firstly".format(csv_path)

        csv_data = pd.read_csv(csv_path)
        data_pickle_path = os.path.join(path, csv_mode + '_data.pickle')
        target_pickle_path = os.path.join(path, csv_mode + '_target.pickle')

        csv_image_name_list = list(csv_data['filename'])
        data_list = []
        for id, image_name in enumerate(csv_image_name_list):
            image_path = os.path.join(os.getcwd(), image_dir, image_name)
            # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
            image = Image.open(image_path).convert('RGB')
            image = transform(image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
            data_list.append(image)
            if (id + 1) % 100 == 0:
                print(f'总共有{len(csv_image_name_list)}张图片，这是第{id + 1}张')
        data = torch.stack(data_list)
        utils.save_as_pickle(data_pickle_path, data)

        # 读取target及其与参数
        csv_image_label_list = list(csv_data['label'])
        target_list = []
        for id, image_label in enumerate(csv_image_label_list):
            target_list.append(image_label)
        targets = torch.from_numpy(np.array(target_list, dtype=np.int32)).long()
        utils.save_as_pickle(target_pickle_path, targets)


def main(root, trsfm, SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    creat_pickle(root, trsfm)


if __name__ == '__main__':
    root = './data/mini-imagenet/tinyImageNet'
    trsfm = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    main(root, trsfm, 123)
