from PIL import Image
from PIL import GifImagePlugin
import torch



def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    longestSide = max(img.size)
    _img = Image.new('RGB', (longestSide, longestSide), (0, 0, 0))
    _img.paste(img, (0, 0))
    _img = _img.resize(size)
    return _img


def keep_image_size_open_gray(path, size=(256, 256)):
    img = Image.open(path)
    longestSide = max(img.size)
    mask = Image.new('1', (longestSide, longestSide))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def gray2RGB(img):
    out_img = torch.cat((img, img, img), 0)
    return out_img
