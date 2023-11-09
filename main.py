from ultralytics import YOLO
import cv2
import os


class LostTHing:
    def __init__(self, video_dir, threshold=0.5):
        self.model = YOLO('yolov8m.pt')
        self.video_dir = video_dir
        self.threshold = threshold
        self.ignore_classes = [56, 26, 62, 0, 41, 66, 63]
        self.person_thing = 0
        self.alarm_frame = None

    # Add things to ignore list
    def add_ignore(self, *args):
        self.ignore_classes.extend(*args)

    # Detect Person,
    # if thing will be in the area, will remember it and when the person exits the frame will issue a warning
    def detect(self):
        cap = cv2.VideoCapture(self.video_dir)
        ret, frame = cap.read()

        self.person_thing = 0
        thing_box = 0

        while ret:
            pers_result = 0
            results = self.model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    if results.names[int(class_id)].upper() == 'PERSON':
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                    if results.names[int(class_id)].upper() == 'PERSON':
                        pers_result = result
                        break
                    else:
                        pers_result = None
            if pers_result is not None:
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result

                    x_1, y_1, x_2, y_2, _score, _class_id = pers_result
                    if x_1 < x1 < x_2 and x_1 < x2 < x_2 and y_1 < y1 < y_2 and y_1 < y1 < y_2 and class_id not in self.ignore_classes:
                        thing = class_id
                        thing_box = result
                        self.person_thing = thing
                        print(f'Обнаружен {results.names[int(self.person_thing)].upper()}')
                        break

            ret, frame = cap.read()

            if self.person_thing != 0 and pers_result is None and ret:
                if thing_box != 0:
                    x1, y1, x2, y2, score, class_id = thing_box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, "the lost thing", (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    print('ТРЕВОГА! ОБНАРУЖЕНА ПОТЕРЯННАЯ ВЕЩЬ!!!')
                    self.alarm_frame = frame
                    cv2.imshow('ALARM', frame)
                    cv2.waitKey(0)
                    cap.release()

        cap.release()
        cv2.destroyAllWindows()

# root_dir = os.path.join('.', 'video')
# video_path = os.path.join(root_dir, 'testing_detect.mp4')
# o = LostTHing(video_path)
# o.detect()
