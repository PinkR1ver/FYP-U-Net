import os
from PIL import Image

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class ImageDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'masks'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segmentName = self.name[index]  ##xx.tif
        segmentPath = os.path.join(self.path, 'masks', segmentName)
        imagePath = os.path.join(self.path, 'imgs', segmentName.replace('_mask.tif', '.tif'))
        segmentImage = keep_image_size_open_gray(segmentPath)
        image = keep_image_size_open(imagePath)
        return transform(image), transform(segmentImage)


def image_location_transfer(rootdir):
    maskSavePath = r'C:\Users\83549\PycharmProjects\FYP\data\masks'
    imgSavePath = r'C:\Users\83549\PycharmProjects\FYP\data\imgs'
    for root, dirs, files in os.walk(os.path.join(rootdir)):
        for file in files:
            if "mask" in file:
                maskPath = os.path.join(root, file)
                print(maskPath)

                img = Image.open(maskPath)
                img.save(os.path.join(maskSavePath, file))
            else:
                imgPath = os.path.join(root, file)
                print(imgPath)

                img = Image.open(imgPath)
                img.save(os.path.join(imgSavePath, file))


if __name__ == '__main__':
    image_location_transfer(r'C:\Users\83549\PycharmProjects\FYP\data\Source')
