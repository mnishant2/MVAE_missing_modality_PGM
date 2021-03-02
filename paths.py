import os

project_dir = "/content/drive/MyDrive/project"
# project_dir = "/content/drive/MyDrive/ift6269/project"
data_dir = os.path.join(project_dir, "data")
rec_data_dir = os.path.join(data_dir, "recordings")
mnist_data_dir = os.path.join(data_dir, "mnist")
speech_data_dir = os.path.join(data_dir, "speech")
lookup_embd_dir = os.path.join(data_dir, "lookup_embd")
OUT_DIR = os.path.join(project_dir, "models")