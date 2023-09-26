import cv2
import numpy as np
import torch
from models.model import Model
from models.models import ConvNextModel

cap = cv2.VideoCapture("2.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

checkpoint = torch.load("./save_weights/model_9.pth")
model.load_state_dict(checkpoint['model'])

m = Model().cuda()
checkpoint = torch.load("./save_weights/model_best.pth")
m.load_state_dict(checkpoint['model'])

cv2.namedWindow("Attention heatmap", cv2.WINDOW_NORMAL)
while cap.isOpened():

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except:
        continue

    # Estimate attention and colorize it
    img = frame.astype(np.float32) / 255
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    heatmap = model(img.to(device)).cpu().detach().numpy()
    h = m(img.to(device)).cpu().detach().numpy()
    color_heatmap = model.draw_heatmap(frame)
    c = m.draw_heatmap(frame)

    combined_img = np.hstack((c, color_heatmap))

    cv2.imshow("Attention heatmap", combined_img)
    # out.write(combined_img)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


