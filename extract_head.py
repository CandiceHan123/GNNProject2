import os
import paddlehub as hub
import cv2

humanseg = hub.Module(name="deeplabv3p_xception65_humanseg")
root_dir = "E:\\dataset\\test1\\"
list = os.listdir(root_dir)

for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != ".jpg":
        continue
    result = humanseg.segmentation(images=[cv2.imread(root_dir + imgName)], visualization=True)
    result.save("E:\\dataset\\test2\\" + imgName)
