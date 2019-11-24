#!/usr/bin/env python3

import torch

MODEL_PATH = 'training_model/'


def perform():
    """
    perform the result of the model
    """
    # load the model
    model = torch.load(MODEL_PATH)

    filename = sys.argv[1]
    cap = cv2.VideoCapture("../../videos/" + filename)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[:,:,1] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.array([frame])
        # pred = model.predict(frame, verbose=1, batch_size=1)
        # pred = np.squeeze(pred, axis = 0)
        # predict based on the trained neural network
        cv2.imshow('frame2',pred)
        cv2.waitKey(1)


if __name__ == "__main__":
        perform()