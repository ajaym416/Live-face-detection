from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import time
import argparse

best_up_to_now = "resnet18_best_face.pth"
model = models.resnet18()
model.fc = torch.nn.Sequential(
    nn.Dropout(0),
    torch.nn.Linear(
        in_features=512,
        out_features=2
    ),
    nn.Sigmoid())
model.load_state_dict(torch.load("resnet18_best_face.pth", map_location=torch.device('cpu')))
model.eval()

eye_model = models.resnet18()
eye_model.fc = torch.nn.Sequential(
    nn.Dropout(0),
    torch.nn.Linear(
        in_features=512,
        out_features=2
    ),
    nn.Sigmoid())
eye_model.load_state_dict(torch.load("eye_classifier_final.pth", map_location=torch.device('cpu')))
eye_model.eval()


class faceDetector():
    def __init__(self, mtcnn, classifier, eye_classifier, show_fps=False, stride=1, video_source=0 , save_video=False , show_all = False):
        self.mtcnn = mtcnn
        self.stride = stride
        self.classifier = classifier
        self.eye_classifier = eye_classifier
        self.show_fps = show_fps
        self.video_source = video_source
        self.save_video =save_video
        self.show_all =show_all

    def _draw(self, frame, boxes, probs, landmarks, eye_lengths):
        """
        Draw a rectangle on the detected face and eyes
        """
        for box, prob, ld, eye_length in zip(boxes, probs, landmarks, eye_lengths):
            # Draw a rectangle on detected face
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]),
                          (255, 0, 0,), thickness=2)
            # Draw landmarks
            #             print(ld[0][0])
            cv2.rectangle(frame, (int(ld[1][0] - eye_length), int(ld[1][1] + eye_length)),
                          (int(ld[1][0] + eye_length), int(ld[1][1] - eye_length)), (0, 140, 255), 1)
            cv2.rectangle(frame, (int(ld[0][0] - eye_length), int(ld[0][1] + eye_length)),
                          (int(ld[0][0] + eye_length), int(ld[0][1] - eye_length)), (0, 140, 255), 1)
        return frame

    def _check_liveness(self, face):
        """
        Check if the detected face is live face or not
        """
        destRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        PILimage = Image.fromarray(destRGB.astype('uint8'), 'RGB')
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        processed_img = preprocess(PILimage)
        batch_t = torch.unsqueeze(processed_img, 0)
        with torch.no_grad():
            out = self.classifier(batch_t)
            _, pred = torch.max(out, 1)
        #             print(out[0][0],out[0][1])
        prediction = np.array(pred[0])
        if prediction == 0:
            return out[0][0], "non_live"
        else:
            return out[0][1], "live_face"

    def _check_eye(self, eye):
        """
        Given the eyes check if the eyes are closed or not
        """
        destRGB = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        PILimage = Image.fromarray(destRGB.astype('uint8'), 'RGB')
        preprocess = transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3473, 0.3473, 0.3473], std=[0.0805, 0.0805, 0.0805])
        ])
        processed_img = preprocess(PILimage)
        batch_t = torch.unsqueeze(processed_img, 0)
        with torch.no_grad():
            out = self.eye_classifier(batch_t)
            _, pred = torch.max(out, 1)
        prediction = np.array(pred[0])
        if prediction == 0:
            #             print("o")
            return ("open_eye")
        elif prediction == 1:
            #             print("c")
            return "closed_eye"

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces  , predict the
            condition of eyes and draw it on frame

        """

        cap = cv2.VideoCapture(self.video_source)
        number_of_frames = 0
        stride_runner = 0
        if not self.save_video:
            out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 8, (640, 480))
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            number_of_frames += 1
            stride_runner += 1
            # detect face box, probability and landmarks
            try:
                if frame.shape[0] < frame.shape[1]:
                    frame = cv2.resize(frame, (640, 480))
                else:
                    frame = cv2.resize(frame, (480, 640))

                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                if not self.show_all:
                    if boxes is None:
                        continue
                eye_lengths = []
                for box, ld in zip(boxes, landmarks):
                    startX, startY, endX, endY = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    eye_length = int(np.sqrt((startX - endX) * (startY - endY)) / 7)
                    eye_lengths.append(eye_length)
                    left_eye_x = int(ld[0][0])
                    left_eye_y = int(ld[0][1])
                    right_eye_x = int(ld[1][0])
                    right_eye_y = int(ld[1][1])
                    left_eye = frame[int(left_eye_y - eye_length):int(left_eye_y + eye_length),
                               int(left_eye_x - eye_length):int(left_eye_x + eye_length)]
                    right_eye = frame[int(right_eye_y - eye_length):int(right_eye_y + eye_length),
                                int(right_eye_x - eye_length):int(right_eye_x + eye_length)]
                    # For the face
                    face = frame[startY:endY, startX:endX]
                    # run the classifier on bounding box
                    if (stride_runner % self.stride == 0) or (stride_runner == 1):
                        prob_value, pred = self._check_liveness(face)
                        pred_left_eye = self._check_eye(left_eye)
                        pred_right_eye = self._check_eye(right_eye)

                    if eye_length >= 15:
                        font_scale = 1
                    else:
                        font_scale = 0.5
                    if pred_left_eye == "open_eye":
                        cv2.putText(frame, "1", (left_eye_x, left_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale - 0.3, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "0", (left_eye_x, left_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale - 0.3, (0, 255, 0), 2, cv2.LINE_AA)
                    if pred_right_eye == "open_eye":
                        cv2.putText(frame, "1", (right_eye_x, right_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale - 0.3, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "0", (right_eye_x, right_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale - 0.3, (0, 255, 0), 2, cv2.LINE_AA)
                    if (pred_left_eye == "closed_eye") and (pred_right_eye == "closed_eye"):
                        cv2.putText(frame, "Eye_closed_non_live", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2,
                                    cv2.LINE_AA)
                        continue

                    if pred == 'live_face':
                        """
                        put the liveness notification in the frame
                        """
                        cv2.putText(frame, "Live_face", (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, str(round(float(prob_value), 4)), (endX, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (0, 255, 0), 2, cv2.LINE_AA)

                    else:
                        if prob_value < 0.65:
                            cv2.putText(frame, "Live_face", (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(frame, str(round(float(prob_value), 4)), (endX, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, "Non_live", (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, str(round(float(prob_value), 4)), (endX, endY), cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, (0, 0, 255), 2, cv2.LINE_AA)

                self._draw(frame, boxes, probs, landmarks, eye_lengths)
            except Exception as e:
                stride_runner = 0
                print(e)

            end_time = time.time()
            number_of_frames += 1
            if end_time-start_time == 0:
                raise Exception("Please select the valid video file")
            avg_frames = number_of_frames / (end_time - start_time)
            if self.show_fps:
                cv2.putText(frame, f"fps:{round(avg_frames, 4)}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
            cv2.putText(frame, "Press 'x' to exit", (-1, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)
            cv2.imshow('Treeleaf AI Challenge 2020', frame)

            #             print(avg_frames)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treeleaf Liveness Detection')
    parser.add_argument("-f", "--showFps", type=bool, default=False, choices=[True, False],
                        help="Choose weather to show the average frame per second or not")
    parser.add_argument("-v", "--videoSource", type=int, default=0, choices=[0, 1, 2],
                        help="Choose the video source to start the liveness detection \
    0: For the primary webcam \
    1: For the secondary webcam \
    2:To select the file from computer")
    parser.add_argument("-p", "--videoPath", type=str, default="demoVideo/demoVideo.avi", help="path to the video ")
    parser.add_argument("-s", "--stride" ,type=int, default=2, choices=[1, 2, 3, 4, 5], help="Increase the FPS by striding")
    parser.add_argument("-c", "--saveVideo" , type = bool , default=False, choices=[True, False] , help= "Set true to save the ouput video")
    parser.add_argument("-d", "--showAll" , type = bool , default=False, choices=[True, False] ,
                        help= "If False it only shows the images in which human faces are present and if True shows all the frames")

    args = vars(parser.parse_args())
    mtcnn = MTCNN()
    if args["showFps"]:
        showFps = True
    else:
        showFps = False
    if args["saveVideo"]:
        saveVideo = True
    else:
        saveVideo = False
    if args["showAll"]:
        showAll = True
    else:
        showAll = False

    videoSource = args["videoSource"]
    videoPath = args["videoPath"]
    stride = args["stride"]
    if videoSource in [0, 1]:
        videoSource = videoSource
    else:
        videoSource = videoPath

    fcd = faceDetector(mtcnn, classifier=model, eye_classifier=eye_model, show_fps=showFps,stride=stride, video_source=videoSource , show_all =showAll )
    fcd.run()