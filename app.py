import cv2
import numpy as np
import mmcv
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout

from kivy.metrics import dp
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout

import os

from inference import load_detection_model, get_detection_prediction
from utils import draw_contours

KV = """
<ImageButton@ButtonBehavior+FitImage>

<ImageManager>
    path: ""
    orientation: "vertical"
    size_hint_y: None
    height: root.height
    padding: dp(10)

    ImageButton:
        source: root.path
        
BoxLayout:
    RecycleView:
        id: rv
        key_viewclass: "viewclass"
        RecycleGridLayout:
            padding: dp(2)
            cols: 3
            default_size: None, dp(48)
            default_size_hint: 1, None
            size_hint_y: None
            height: self.minimum_height
"""

predictor = load_detection_model()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    image = ObjectProperty(None)
    available_image_format = ['.png', '.jpg', '.jpeg']
    dirr = "/Users/terekhin/walrus_hack/submission/"
    manager_list = []

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        GalleryApp().run()
        
    def load(self, path, filename):
        for _, img in enumerate(filename):
            image = mmcv.imread(os.path.join(path, img))
            result = get_detection_prediction(predictor, image)
            count_class = draw_contours(image, result)
            self.cv_image = image.astype(np.uint8)
            cv2.imwrite(os.path.join("/Users/terekhin/habarovsk_hack/submission/", f"detected_{count_class}.jpg"), image)
            self.dismiss_popup()

class ImageManager(BoxLayout):
    pass


class GalleryApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.manager_list = []
        self.dir = "/Users/terekhin/habarovsk_hack/submission/"
        self.available_image_format = ['.png', '.jpg', '.jpeg']  # etc

    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        self.load_images()

    def load_images(self):
        if not self.manager_list:
            for image in os.listdir(self.dir):
                target_filename, target_file_extension = os.path.splitext(image)
                if target_file_extension in self.available_image_format:
                    path_to_image = os.path.join(self.dir, image)
                    self.manager_list.append(
                        {
                            "viewclass": "ImageManager",
                            "path": path_to_image,
                            "height": dp(200),
                        }
                    )
            self.root.ids.rv.data = self.manager_list

class Editor(MDApp):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)


if __name__ == '__main__':
    Editor().run()