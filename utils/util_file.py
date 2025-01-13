import os
import shutil


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


def list_files_in_current_directory(dir, extension='.pcd'):
    pcd_files = []
    for item in os.listdir(dir):
        full_path = os.path.join(dir, item)
        if os.path.isfile(full_path) and item.endswith(extension):
            pcd_files.append(full_path)
    return pcd_files


def mkdir_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def delete_folder(folder_path):
    """删除指定路径的文件夹及其内容"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"文件夹 '{folder_path}' 已被删除。")
    else:
        print(f"文件夹 '{folder_path}' 不存在。")


def list_directories_in_current_path():
    """遍历当前目录下的文件夹并打印其名称"""
    current_path = os.getcwd()  # 获取当前工作目录
    print(f"当前目录: {current_path}")

    # 遍历当前目录
    for item in os.listdir(current_path):
        item_path = os.path.join(current_path, item)  # 获取完整路径
        if os.path.isdir(item_path):  # 检查是否为文件夹
            print(f"文件夹: {item}")


if __name__ == '__main__':
    absolute_path = "/path/to/your/directory/"

    # prepare_directory(absolute_path)
    print(os.path.isdir(absolute_path))
