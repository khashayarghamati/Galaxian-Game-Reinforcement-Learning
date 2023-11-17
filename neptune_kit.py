import neptune


class Neptune:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Neptune, cls).__new__(cls)
        return cls.instance

    def init(self):
        self.run = neptune.init_run(
            project="khashayar/GalaxianRL",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YzhjYmU2Zi03ODBjLTQ4YjEtODAzMy1lOTJhN2Q1YWU1YjUifQ==",
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
