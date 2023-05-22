#比较两张图片的相似度,返回值为0-1之间的浮点数，越接近1，说明两张图片越相似
from PIL import Image
from difflib import SequenceMatcher

def calc_similar(li1, li2):
    return SequenceMatcher(None, li1, li2).quick_ratio()

if __name__ == '__main__':
    img1 = Image.open('test_images_second/test_0copy.jpg')
    img2 = Image.open('test_images_third/test_0copy_hidden.png')
    h1 = img1.histogram()
    h2 = img2.histogram()
    print(calc_similar(h1, h2))
