import os
import re

func_start_with = ['void', 'int', 'float', 'double',
                   'char', 'bool', 'struct', 'class', 'enum']


def find_cpp_files(directory):
    cpp_files = []
    h_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.cc'):
                cpp_files.append(os.path.join(root, file))
            elif file.endswith('.h') or file.endswith('.hpp'):
                h_files.append(os.path.join(root, file))
    return cpp_files, h_files


def filter_lock(line):
    if line.startswith('std::lock_guard<std::mutex>') or line.startswith('std::unique_lock<std::mutex>') or \
            line.startswith('std::shared_lock<std::shared_mutex>') or line.startswith('lock_guard<mutex>') or \
            line.startswith('unique_lock<mutex>') or line.startswith('shared_lock<shared_mutex>') or \
            line.startswith('std::lock_guard<mutex>') or line.startswith('std::unique_lock<mutex>') or \
            line.startswith('std::shared_lock<shared_mutex>') or line.startswith('lock_guard<std::mutex>') or \
            line.startswith('unique_lock<std::mutex>') or line.startswith('shared_lock<std::shared_mutex>'):
        return True
    return False


def extract_functions_from_include(file_content):
    # return_type func_name(args);
    define_function_pattern = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\-\_]*)\s*\(\s*([\w\s\=\"\-\_\::\<\>\&\,]*)\s*\)\s*\;$'
    # return_type func_name(args){}
    function_pattern_1 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{\s*([\w\s\;\=\.\-\_\::\<\>\&\,]*)\s*\}$'
    # return_type func_name(args){
    function_pattern_2 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{$'

    functions = {}

    pre = ''
    func_name = "{return_type} {func_name} ({args})"
    functions['others'] = {'set': set(), 'list': []}
    is_func_content = 0
    for line in file_content:
        if is_func_content > 0:
            if line.startswith('}') and line.endswith('{'):
                continue
            elif line.endswith('{'):
                is_func_content += 1
            elif line.startswith('}'):
                is_func_content -= 1
            continue
        if line.startswith('static'):
            line = line.split('static')[1].strip()
        if line.startswith('inline'):
            line = line.split('inline')[1].strip()
        if line.startswith('class') or line.startswith('struct'):
            pre = line.split(' ')[1]
            pre = pre.split('{')[0]
            pre = pre.split(':')[0]
            functions[pre] = {'set': set(), 'list': []}
            continue
        elif line.startswith('enum'):
            continue
        elif pre != '' and line.startswith('};'):
            pre = ''
            continue
        elif filter_lock(line):
            continue

        matches = re.search(define_function_pattern, line)
        function_name = ''
        if not matches:
            matches = re.search(function_pattern_2, line)
            if not matches:
                matches = re.search(function_pattern_1, line)
            is_func_content = True

        if matches and matches.group(2) != '':
            function_name = func_name.format(
                return_type=matches.group(1),
                func_name=matches.group(2),
                args=matches.group(3)
            )
            if pre == '':
                key = 'others'
            else:
                key = pre
            functions[key]['set'].add(function_name)
            functions[key]['list'].append(matches.group(2))

    return functions


def extract_functions_from_cpp(file_content, functions):
    # return_type func_name(args){}
    function_pattern_1 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{\s*([\w\s\;\=\.\-\_\::\<\>\&\,]*)\s*\}$'
    # return_type func_name(args){
    function_pattern_2 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{$'
    # return_type class::func_name(args){}
    function_pattern_3 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\-\_]*)\::([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{\s*([\w\s\;\=\.\-\_\::\<\>\&\,]*)\s*\}$'
    # return_type class::func_name(args){
    function_pattern_4 = r'^([\w\=\.\-\_\::\<\>]*)\s*([\w\-\_]*)\::([\w\=\.\-\_\::\<\>]*)\s*\(\s*([\w\s\=\.\-\_\::\<\>\&\,]*)\s*\)\s*\{$'

    func_name = "{return_type} {func_name} ({args})"
    key = 'others'
    for line in file_content:
        matches = re.search(function_pattern_1, line)
        if matches:
            return_type = matches.group(1)
            func_name = matches.group(2)
            args = matches.group(3)
        else:
            matches = re.search(function_pattern_2, line)
            if matches:
                return_type = matches.group(1)
                func_name = matches.group(2)
                args = matches.group(3)
            else:
                matches = re.search(function_pattern_3, line)
                if matches:
                    return_type = matches.group(1)
                    key = matches.group(2)
                    func_name = matches.group(3)
                    args = matches.group(4)
                else:
                    matches = re.search(function_pattern_4, line)
                    if matches:
                        return_type = matches.group(1)
                        key = matches.group(2)
                        func_name = matches.group(3)
                        args = matches.group(4)
        if matches and func_name != '':
            if key not in functions:
                print(key)
            else:
                if func_name not in functions[key]['list']:
                    function_name = func_name.format(
                        return_type=return_type,
                        func_name=func_name,
                        args=args
                    )
                    functions[key]['set'].add(function_name)
                    functions[key]['list'].append(func_name)
    return functions


def find_relate_include(cpp_file, functions):
    file_name = os.path.basename(cpp_file)
    file_name = file_name.split('.')[0]
    if file_name in functions:
        return file_name
    else:
        with open(cpp_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            for line in file_content.splitlines():
                line = line.strip()
                if line.startswith('#include'):
                    tmp = line.split(' "')
                    if len(tmp) == 2:
                        include_file = line.split(' "')[1]
                        include_file = include_file.split('.')[0]
                        if include_file in functions:
                            return include_file
        return None


def analyze_project(directory):
    cpp_files, h_files = find_cpp_files(directory)
    all_functions = {}
    call_graph = {}

    for file in h_files:
        file_name = os.path.basename(file)
        file_name = file_name.split('.')[0]
        with open(file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            code_format = reformat_code(file_content)
            functions = extract_functions_from_include(code_format)
            all_functions[file_name] = functions
    for file in cpp_files:
        file_name = find_relate_include(file, all_functions)
        if not file_name:
            print(f'Cannot find related include file for {file}')
            continue
        with open(file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            code_format = reformat_code(file_content)
            functions = extract_functions_from_cpp(
                code_format, all_functions[file_name])
            all_functions[file_name] = functions

    return all_functions, call_graph


def reformat_code(file_content):
    code_lines = []
    pre_code = ''
    skip_start = 0
    for line in file_content.splitlines():
        line = line.strip()
        if skip_start == 1:
            if line.startswith('*/') or line.startswith('///'):
                skip_start = 0
            continue
        elif line.startswith('#') or line.startswith('//') or line.startswith('using'):
            continue
        elif line.startswith('/*') or line.startswith('///'):
            skip_start = 1
            continue
        elif line.endswith(';') or line.endswith('}'):
            code_lines.append(pre_code + line)
            pre_code = ''
        elif line == '':
            continue
        elif line.endswith('{'):
            code_lines.append(pre_code + line)
            pre_code = ''
        else:
            pre_code += line + ' '
    return code_lines


if __name__ == '__main__':
    path = '/home/demo/Documents/code/infrastructure4.0/omnisense4.0/common/'
    functions, call_graph = analyze_project(path)

    print("所有函数:")
    for func in functions:
        print(func)

    print("\n函数调用关系:")
    for caller, callee_list in call_graph.items():
        for callee in callee_list:
            print(f"{caller} -> {callee}")
