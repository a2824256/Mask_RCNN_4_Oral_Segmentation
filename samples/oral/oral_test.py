from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('Qt5Agg')
file = "../../oral_dataset/annotations.json"
pic_path = "../../oral_dataset/JPEGImages/0.jpg"
im = cv2.imread(pic_path)
plt.imshow(im); plt.axis('off')
cc = COCO(file)
annIds = cc.getAnnIds(imgIds=0)
anns = cc.loadAnns(annIds)
cc.showAnns(anns)
plt.show()