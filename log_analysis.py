import xml.etree.ElementTree as ET
import os
import re
import importlib
import csv


package_name = 'pandas'
if importlib.util.find_spec(package_name) is not None:
    import pandas as pd
    PANDAS_USEFUL = True
else:
    PANDAS_USEFUL = False

PANDAS_USEFUL = False


class LogFromMainboard:
    def __init__(self, file_path=[]):
        self._init_conf()
        self._init_data()
        self._init_func()
        if file_path:
            self.set_file_path(file_path)

    def _init_func(self):
        if PANDAS_USEFUL:
            self._decode_log = self._decode_log_df
            self._to_csv = self._logs_df_csv
        else:
            self._decode_log = self._decode_log_set
            self._to_csv = self._logs_set_csv

    def set_file_path_by_dir(self, log_dir, filter_str='', use_mainboard=False):
        self._file_path = []
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path) and '.INFO' in filename and not filename.endswith('.json'):
                if 'mainboard' in filename:
                    if use_mainboard:
                        self._mainboard_file_path.append(file_path)
                else:
                    if filter_str != '' and filter_str in filename:
                        continue
                    self._file_path.append(file_path)

    def set_file_path(self, file_path):
        if isinstance(file_path, list):
            self._file_path = file_path
        else:
            raise ValueError("file_path must be a list.")

    def decode_logs(self, file_path=None):
        if file_path:
            self.set_file_path(file_path)
        for log_path in self._file_path:
            self._decode_log(log_path, self._log_df)

        for log_path in self._mainboard_file_path:
            self._decode_log(log_path, self._mainboard_log)

        if PANDAS_USEFUL:
            self._log_df = self._log_df.drop_duplicates(subset=self._columns)
            self._log_df = self._log_df.sort_values(by=self._sort_columns)
            self._mainboard_log = self._mainboard_log.drop_duplicates(
                subset=self._columns)
            self._mainboard_log = self._mainboard_log.sort_values(
                by=self._sort_columns)

    @property
    def logs_df(self):
        return self._log_df.copy()

    @property
    def mainboard_logs_df(self):
        return self._mainboard_log.copy()

    def logs_to_csv(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path_log = os.path.abspath(os.path.join(out_dir, 'log.csv'))
        out_path_mainboard_log = os.path.abspath(
            os.path.join(out_dir, 'mainboard_log.csv'))
        self._to_csv(out_path_log, out_path_mainboard_log)

    def _logs_set_csv(self, out_path_log, out_path_mainboard_log):
        try:
            with open(out_path_log, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(tuple(self._columns))
                for tup in self._log_df:
                    writer.writerow(tup)
            with open(out_path_mainboard_log, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(tuple(self._columns))
                for tup in self._mainboard_log:
                    writer.writerow(tup)
        except Exception as e:
            print(f"Error while writing log to csv: {e}")

    def _logs_df_csv(self, out_path_log, out_path_mainboard_log):
        print(self._log_df.shape)
        try:
            self._log_df.to_csv(out_path_log, index=False)
            self._mainboard_log.to_csv(out_path_mainboard_log, index=False)
        except Exception as e:
            print(f"Error while writing log to csv: {e}")

    def _init_conf(self):
        self._file_path = []
        self._mainboard_file_path = []
        self._columns = ['log_level', 'mmdd', 'hh', 'mm',
                         'ss.uuuuuu', 'threadid', 'file', 'line', 'message']
        self._match_pattern = r'([EIWFA])(\d{4}) (\d{2}:\d{2}:\d{2}\.\d{6})\s+(\d+)\s+([\w\.]+:\d+)] (.*?)$'
        self._match_num = 6
        self._sort_columns = ['mmdd', 'hh', 'mm', 'ss.uuuuuu']
        self._add_log_files = []
        self._add_mainboard_files = []

    def _init_data(self):
        if PANDAS_USEFUL:
            self._log_df = pd.DataFrame(columns=self._columns)
            self._mainboard_log = pd.DataFrame(columns=self._columns)
        else:
            self._log_df = set()
            self._mainboard_log = set()

    def _decode_log_df(self, log_path, df):
        with open(log_path, 'r') as file:
            for line in file:
                line = line.strip()
                matches = re.search(self._match_pattern, line)
                if matches and matches.lastindex == self._match_num:
                    match_t_split = matches.group(3).split(':')
                    match_file_split = matches.group(5).split(':')
                    df.loc[len(df)] = {
                        'log_level': matches.group(1),
                        'mmdd': matches.group(2),
                        'hh': match_t_split[0],
                        'mm': match_t_split[1],
                        'ss.uuuuuu': match_t_split[2],
                        'threadid': matches.group(4),
                        'file': match_file_split[0],
                        'line': match_file_split[1],
                        'message': matches.group(6).strip()
                    }
                elif line.startswith('E1') or line.startswith('E0'):
                    print('match error: ', line)

    def _decode_log_set(self, log_path, data):
        with open(log_path, 'r') as file:
            for line in file:
                line = line.strip()
                matches = re.search(self._match_pattern, line)
                if matches and matches.lastindex == self._match_num:
                    match_t_split = matches.group(3).split(':')
                    match_file_split = matches.group(5).split(':')
                    add_data = (matches.group(1), matches.group(2), match_t_split[0],
                                match_t_split[1], match_t_split[2],
                                matches.group(4),
                                match_file_split[0], match_file_split[1],
                                matches.group(6).strip())
                    data.add(add_data)
                elif line.startswith('E1') or line.startswith('E0'):
                    print('match error: ', line)

    def add_file_path(self, file_path: list):
        if not isinstance(file_path, list):
            raise ValueError("file_path must be a list.")
        for file in file_path:
            if 'mainboard' in file:
                self._mainboard_file_path.append(file)
                self._add_mainboard_files.append(file)
            else:
                self._file_path.append(file)
                self._add_files.append(file)

    def decode_new_logs(self):
        for log_path in self._add_files:
            self._decode_log(log_path, self._log_df)
        self._add_files = []
        for log_path in self._add_mainboard_files:
            self._decode_log_df(log_path, self._mainboard_log)
        self._add_mainboard_files = []
        if PANDAS_USEFUL:
            self._log_df = self._log_df.drop_duplicates(subset=self._columns)
            self._log_df = self._log_df.sort_values(by=self._sort_columns)
            self._mainboard_log = self._mainboard_log.drop_duplicates(
                subset=self._columns)
            self._mainboard_log = self._mainboard_log.sort_values(
                by=self._sort_columns)

    def add_logs_by_file_pathes(self, file_paths: list):
        self.add_file_path(file_paths)
        self.decode_new_logs()

    def clear(self):
        self._init_conf()
        self._init_data()

    def clear_df(self):
        self._init_data()


def class_name_from_dag(file_path):
    name = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            matches = re.findall(r'^name : "(.*?)"', line)
            if matches:
                name.append(matches[0])
    return name


def read_launch(file_path):
    root_dir = file_path.split('launch')[0]
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 提取所有节点信息
    module_info = {}
    process_info = {}

    for module in root.findall('module'):
        module_name = module.find('name').text
        dag_conf = module.find('dag_conf').text if module.find(
            'dag_conf') is not None else ''
        process_name = module.find('process_name').text
        exception_handler = module.find('exception_handler').text
        if module_name in module_info:
            if 'param_server' in process_name:
                module_name = 'param_server'
            else:
                raise Exception(f"Duplicate module name: {module_name}")
        class_name = []
        if dag_conf:
            conf_path = root_dir + 'launch' + dag_conf.split('launch')[1]
            class_name = class_name_from_dag(conf_path)
        module_info[module_name] = {
            'dag_conf': dag_conf,
            'process_name': process_name,
            'exception_handler': exception_handler,
            'class_name': class_name
        }
        if process_name in process_info:
            process_info[process_name].append(module_name)
        else:
            process_info[process_name] = [module_name]

    return module_info, process_info


def get_log_files(log_dir):
    files = []
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path) and not filename.endswith('.json'):
            files.append(filename)


if __name__ == '__main__':
    launch_file = "/home/demo/Documents/Highway/install/od/SW/modules/omnisense/launch/current.launch"
    log_dir = "/home/demo/Documents/Highway/install/od/SW/data/log/"
    log_c = LogFromMainboard()
    log_c.set_file_path_by_dir(log_dir)
    log_c.decode_logs()
    log_c.logs_to_csv('./')
    print('OK')
