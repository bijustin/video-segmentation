#!/usr/bin/env python3

from keras.models import load_model


def perform(idx):
    """
    perform the result of the model
    """
    # load the model
    model = load_model('trained_model/model_{}.h5'.format(idx))

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
        pred = model.predict(frame, verbose=1, batch_size=1)
        pred = np.squeeze(pred, axis = 0)
        cv2.imshow('frame2',pred)
        cv2.waitKey(1)


if __name__ == "__main__":
        perform(0)