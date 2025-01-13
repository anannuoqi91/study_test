import logging
import time
import os


class UtilLogger():
    def __init__(self, name, path=None, level=logging.INFO):
        self._logger = logging.getLogger(name)
        if self._logger.handlers:
            return
        self._level = level

        self._file_path = self._init_path(path)
        self._formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        self._init_logger()

    def _init_path(self, path):
        out_path = path
        if path is None:
            file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            out_path = f"{file_name}.log"
        else:
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
        return out_path

    def _init_logger(self):
        self._logger.setLevel(self._level)

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self._file_path)

        console_handler.setFormatter(self._formatter)
        file_handler.setFormatter(self._formatter)

        # 添加处理器到根 logger
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    @property
    def logger(self):
        return self._logger


if __name__ == "__main__":
    logger = UtilLogger("test_logger")
    logger.log("This is a info log", logging.DEBUG)
