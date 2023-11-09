import unittest
import cv2
from main import LostTHing
import os

VIDEOS_DIR = os.path.join('.', 'video')
video_path = os.path.join(VIDEOS_DIR, 'testing_detect.mp4')
img_path = os.path.join(VIDEOS_DIR, 'test_img.png')


class TestVideoMethods(unittest.TestCase):
    def test_video(self):
        cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()

        self.assertTrue(ret)

    def test_frame(self):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        self.assertEqual(frame[0][0][0], 254)

    def test_predict(self):
        img = cv2.imread(img_path)
        test_instance = LostTHing(img_path)
        results = test_instance.model(img)[0]
        classes = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            classes.append(results.names[int(class_id)].upper())
        self.assertIn('PERSON', classes)

    def test_detect(self):
        o = LostTHing(video_path)
        o.detect()
        self.assertEqual(o.person_thing, 24)
        self.assertIsNotNone(o.alarm_frame)


if __name__ == '__main__':
    unittest.main()
