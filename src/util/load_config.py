import yaml
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config.yaml")) as infile:
	config = yaml.safe_load(infile)
