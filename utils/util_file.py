import os


def prepare_directory(path):
    if not os.path.exists(path):
        if path.endswith(os.path.sep):
            os.makedirs(path)
        else:
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)


if __name__ == '__main__':
    absolute_path = "/path/to/your/directory/"

    # prepare_directory(absolute_path)
    print(os.path.isdir(absolute_path))
