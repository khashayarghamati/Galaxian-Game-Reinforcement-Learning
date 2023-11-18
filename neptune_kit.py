import os

import neptune
from dotenv import load_dotenv


class Neptune:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Neptune, cls).__new__(cls)
        return cls.instance

    def init(self):
        load_dotenv()
        self.run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_TOKEN"),
        )
        print("Neptune initialized")

    def save_chart(self, name, data):
        if not hasattr(self, 'run'):
            self.init()

        self.run[name].append(data)

    def save_figure(self, name, image):
        if not hasattr(self, 'run'):
            self.init()

        self.run[name].upload(image)
