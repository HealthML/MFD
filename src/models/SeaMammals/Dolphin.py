from models.SeaMammals.SeaMammal import SeaMammal

#class Dolphin(SeaMammal):
#    model_dict = {
#        "cnn": [
#            {"conv":{"out_channels": 64, "stride": 1, "kernel_size": 3, "padding": 0}, "max": {"kernel_size": 4, "stride": 1, "padding": 0}},
#        ],
#        "add_linears": [4000]
#    }
#

class Dolphin(SeaMammal):
    model_dict = {
        "cnn": [
            {"conv":{"out_channels": 32, "stride": 1, "kernel_size": 3, "padding": 0}, "max": {"kernel_size": 4, "stride": 1, "padding": 0}},
            {"conv":{"out_channels": 64, "stride": 1, "kernel_size": 3, "padding": 0}, "max": {"kernel_size": 4, "stride": 2, "padding": 0}},
            {"conv":{"out_channels": 128, "stride": 1, "kernel_size": 3, "padding": 0}, "max": {"kernel_size": 4, "stride": 2, "padding": 0}},
        ],
        "add_linears": [4000]
    }
