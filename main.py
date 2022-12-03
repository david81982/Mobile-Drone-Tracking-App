# ------------------------------------------------------------------------------------
# Author: David Gomez
# University: Texas Tech University
#
# Description: Create a mobile application that can be used for drone detection
#   and tracking.
#
# Version 3.1.1
#
# Notes:
#
# References: https://github.com/kivy/kivy
#  https://appdividend.com/2022/10/18/python-cv2-videocapture/
# ------------------------------------------------------------------------------------
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty
from core import Core
import cv2
import os


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

Builder.load_string('''
<Camera_Test>:
    orientation: 'vertical'
    theme: "Dark"

    Button:
        text: 'Show the tracking results'
        on_press: root.tracking()
        size_hint_y: None
        height: '48dp'
        color: "white"
        background_color: (1,0,0,1)
        italic: True

''')


class Camera_Test(BoxLayout):

    filename = StringProperty('Good2.avi')
    frames_per_second = NumericProperty(60.0)
    video_resolution = StringProperty('720p')

    def __init__(self, **kwargs):
        super(Camera_Test, self).__init__(**kwargs)
        self.img1 = Image()
        self.add_widget(self.img1)
        self.capture = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter(self.filename, self.get_video_type(self.filename), self.frames_per_second,
                                   self.get_dims(self.capture, self.video_resolution))
        Clock.schedule_interval(self.update, 1 / self.frames_per_second)

    def update(self, *args):
        ret, frame = self.capture.read()
        self.out.write(frame)
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.img1.texture = texture

    def change_resolution(self, cap, width, height):
        self.capture.set(3, width)
        self.capture.set(4, height)

    # grab resolution dimensions and set video capture to it.
    def get_dims(self, cap, video_resolution='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if self.video_resolution in STD_DIMENSIONS:
            width, height = STD_DIMENSIONS[self.video_resolution]

        self.change_resolution(cap, width, height)
        return width, height

    def get_video_type(self, filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']



    def tracking(self):
        video_path = "Good2.avi"
        is_frame_skip = True
        skip_frame = 1

        c = Core("/resnet50_csv_24_inference.h5")
        c.set_model(c.get_model())
        cap = cv2.VideoCapture(video_path)

        count = 0
        n = 0

        savenum = 1

        while (cap.isOpened()):
            # Capture each frame
            count += 1
            ret, image = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            drawing_image = c.get_drawing_image(image)

            if is_frame_skip:
                if count % skip_frame == 0:
                    processed_image, scale = c.pre_process_image(image)
                    boxes, scores, labels = c.predict(c.model, processed_image, scale)

                    detections = c.draw_boxes_in_image(drawing_image, boxes, scores)
            else:
                processed_image, scale = c.pre_process_image(image)
                boxes, scores, labels = c.predict(c.model, processed_image, scale)

                detections = c.draw_boxes_in_image(drawing_image, boxes, scores)

            cv2.imshow('Frame', cv2.resize(drawing_image, (0, 0), fx=0.5, fy=0.5))

            # Press Q on keyboard to exit
            # if savenum == 1000:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class DroneTracker(App):

    def build(self):
        return Camera_Test()


if __name__ == '__main__':
    DroneTracker().run()
