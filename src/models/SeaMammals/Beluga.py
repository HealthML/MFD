from models.SeaMammals.SeaMammal import SeaMammal

class Beluga(SeaMammal):
    model_dict = {
        "cnn": [
            {"conv":{"out_channels": 320, "stride": 1, "kernel_size": 8, "padding": 0}},
            {"conv":{"out_channels": 320, "stride": 1, "kernel_size": 8, "padding": 0}, "dropout": 0.2, "max": {"kernel_size": 4, "stride": 4, "padding": 0}},
            {"conv":{"out_channels": 480, "stride": 1, "kernel_size": 8, "padding": 0}},
            {"conv":{"out_channels": 480, "stride": 1, "kernel_size": 8, "padding": 0}, "dropout": 0.2, "max": {"kernel_size": 4, "stride": 4, "padding": 0}},
            {"conv":{"out_channels": 640, "stride": 1, "kernel_size": 8, "padding": 0}},
            {"conv":{"out_channels": 640, "stride": 1, "kernel_size": 8, "padding": 0}, "dropout": 0.5},
        ],
        "add_linears": [2003]
    }
