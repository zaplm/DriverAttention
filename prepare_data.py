import cv2
import os


def main():
    train_path = "./BDDA/training"
    val_path = "./BDDA/validation"

    video_path = val_path + "/camera_videos"
    gazemap_path = val_path + "/gazemap_videos"
    videos = os.listdir(video_path)

    for v in videos:
        video = os.path.join(video_path, v)
        gaze = v[:-4] + "_pure_hm.mp4"
        gaze = os.path.join(gazemap_path, gaze)

        train_data = read_video(video)
        gt_data = read_video(gaze)

        while len(train_data) > len(gt_data):
            train_data.pop()
        while len(gt_data) > len(train_data):
            gt_data.pop()

        for i in range(len(train_data)):
            print("{}_{}.jpg".format(v[:-4], i + 1))
            cv2.imwrite("./dataset/val/camera/{}_{}.jpg".format(v[:-4], i + 1), train_data[i])
            gt_data[i] = gt_data[i][96:672, :, :]
            gt_data[i] = cv2.cvtColor(gt_data[i], cv2.COLOR_BGR2GRAY)
            cv2.imwrite("./dataset/val/gaze/{}_{}.jpg".format(v[:-4], i + 1), gt_data[i])
            # x = cv2.imread("./dataset/train/gaze/{}_{}.jpg".format(v[:-4], i + 1), cv2.IMREAD_GRAYSCALE)


def read_video(path):
    x = []
    reader = cv2.VideoCapture(path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    temp = int(fps / 3) + 1
    count = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        count += 1
        if count % temp == 0:
            x.append(frame)

    return x


if __name__ == '__main__':
    main()
