import psutil
import time
import pandas as pd
import matplotlib.pyplot as plt


def get_pid_by_keyword(keyword):
    pid = []
    name = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if keyword in proc.info['name']:
            pid.append(proc.info['pid'])
            name.append(proc.info['cmdline'][3])
    print(pid)
    print(name)
    return pid, name


def log_process_info(log_file, pid_list):
    out = []
    is_running = len(pid_list)
    while is_running:
        is_running = 0
        # 获取所有进程列表
        for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'memory_percent', 'memory_info']):
            if proc.info['pid'] in pid_list:
                pid = proc.info['pid']
                name = proc.info['cmdline'][3]
                cpu_percent = proc.info['cpu_percent']
                mem_percent = proc.info['memory_percent']
                memory_info = round(proc.info['memory_info'].rss /
                                    (1024 * 1024 * 1024), 2)
                out.append([pid, name, cpu_percent, mem_percent, memory_info])
                is_running += 1
        # 间隔时间（秒）
        time.sleep(60 * 10)
    out = pd.DataFrame(out, columns=[
                       'PID', 'Name', 'CPU Usage (%)', 'Memory Usage (%)', 'Memory Info'])
    out.to_csv(log_file, index=False)
    return out


def plan(data, pid_l, pid_name, out_path):
    fig, axs = plt.subplots(
        len(pid_l), 1, figsize=(3*len(pid_l), 4*len(pid_l)))
    title = 'cpu mem analysis'
    bz = -1
    for i in range(len(pid_l)):
        pid = pid_l[i]
        bz = bz + 1
        tmp_df = data[data['PID'] == pid]
        x = [i for i in range(len(tmp_df))]
        cpu = tmp_df['CPU Usage (%)'].tolist()
        mem = tmp_df['Memory Usage (%)'].tolist()
        res = tmp_df['Memory Info'].tolist()
        line1 = axs[bz].plot(x, cpu, 'k-', label='CPU Usage (%)')
        line2 = axs[bz].plot(x, mem, 'r--', label='Memory Usage (%)')
        axs[bz].set_title(pid_name[i])
        axs[bz].set_xlabel('time')
        axs[bz].set_ylabel('percent %')

        ax2 = axs[bz].twinx()
        ax2.set_ylabel('RES G')
        line3 = ax2.plot(x, res, 'b--', label='RES (G)')
        ax2.tick_params(axis='y')

        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        axs[bz].legend(lines, labels, loc='upper right')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    keyword = 'mainboard'
    log_file = "process_monitor.csv"
    pid_l, pid_name = get_pid_by_keyword(keyword)
    data = log_process_info(log_file, pid_l)
    # data = pd.read_csv(log_file)
    plan(data, pid_l, pid_name, "process_monitor.png")
