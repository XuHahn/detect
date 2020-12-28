from yoloV3 import *
from line import *
import sys
import os
img = cv2.imread('test.jpg')
model = YOLO_V3()
model.load_net_weights()


def draw_img(img):
    lines = lines_detect(img)
    line_img = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    img = model.detect(img, model, False)
    result = cv2.addWeighted(img, 1, line_img, 0.95, 1)
    cv2.imwrite('static/images/aaa.jpg',result)
    # path = os.path.abspath('aaa.jpg')
    return "aaa.jpg"


def draw_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('D:\predict\predict.avi', fourcc, fps, size)

    ret = True
    while ret:
        ret, frame = cap.read()
        if frame is None:
            break
        lines = lines_detect(frame)
        line_img = np.zeros_like(frame)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

        img = model.detect(frame, model, False)
        result = cv2.addWeighted(img, 1, line_img, 0.95, 1)
        cv2.imshow('result', result)
        cv2.waitKey(1)
        out.write(result)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # draw_video(r'D:\vedio\mda-ke1ujzduvrzw9ra5.mp4')
    args = sys.argv
    img = cv2.imread(args[1])
    print(type(img))
    img = draw_img(img)
    cv2.imshow('img',img)
    cv2.waitKey()
