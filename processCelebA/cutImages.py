"""这段代码要在数据集图片同一目录下执行"""
"""将图片裁切为64*64的正方形"""

from PIL import Image
import face_recognition
import os
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96))
])

list = os.listdir("E:\\dataset\\img_align_celeba\\img_align_celeba")
for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != ".jpg":
        continue
    image = face_recognition.load_image_file(imgName)
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        width = right - left
        height = bottom - top
        if width > height:
            right -= (width-height)
        elif width < height:
            bottom -= (height-width)
        face_image = image[top:bottom, left:right]
        resized_img = transform(face_image)
        resized_img.save("E:\\dataset\\img_celebA_resize\\"+imgName)
