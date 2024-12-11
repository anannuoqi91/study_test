import os


def prepare_directory(path):
    if not os.path.exists(path):
        if path.endswith(os.path.sep):
            os.makedirs(path)
        else:
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)


def list_files_with_filter(directory, filter_str):
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and not filename.endswith(filter_str):
            files.append(filename)

    return files


def mkdir_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    absolute_path = "/path/to/your/directory/"

    # prepare_directory(absolute_path)
    print(os.path.isdir(absolute_path))
