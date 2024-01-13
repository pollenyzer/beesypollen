import cv2
from os.path import join
import json
from pollenyzer import Pollenyzer

img = cv2.imread(join("resources", "test_img.jpg"))

res = Pollenyzer()(img)

# pretty print
print(json.dumps(res, indent=2))
