import datetime
import time
import subprocess
from huggingface_hub import hf_hub_url
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import filter_repo_objects
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description='Args of downloading')
# Experiment Args
parser.add_argument('--repo_id', type=str)
parser.add_argument('--save_path',default=None, type=str)
parser.add_argument('--token',default=None, type=str)

args = parser.parse_args()


thread_num=2

repo_id = args.repo_id
save_path = args.save_path
token = args.token

# 执行命令
def execCmd(cmd):
    try_num = 0
    command = ' '.join(cmd)
    while True:
        print('[{}]开始第{}执行命令：{}'.format(datetime.datetime.now(),try_num,command))
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        returncode =  process.poll()
        print("[{}]第{}执行命令{}返回码：{}".format(datetime.datetime.now(),try_num,command,returncode))
        if returncode==0:
            break
        time.sleep(1)
        try_num+=1


if __name__ == '__main__':
    
    # 获取项目信息
    while True:
        try:
            _api = HfApi()
            repo_info = _api.repo_info(
                repo_id=repo_id,
                repo_type="model",
                revision='main',
                token=token,
            )

            # 获取文件信息
            filtered_repo_files = list(
                filter_repo_objects(
                    items=[f.rfilename for f in repo_info.siblings],
                    allow_patterns=None,
                    ignore_patterns=None,
                )
            )
            break
        except Exception as err:
            print('获取下载链接失败：{}'.format(err))
            time.sleep(2)

    cmds = []

    # 需要执行的命令列表
    for file in filtered_repo_files:
        # 获取路径
        url =hf_hub_url(repo_id=repo_id, filename=file) #"https://huggingface.co/google/gemma-2-2b/tree/main"#
        # 断点下载指令
        cmds.append(['wget','-T','60','-c',url,'-P',save_path,'--no-check-certificate'])

    print("程序开始%s" % datetime.datetime.now())
    with ThreadPoolExecutor(max_workers=thread_num) as t: 
        all_task = [t.submit(execCmd, cmd) for cmd in cmds]
        finish_num = 0
        for future in as_completed(all_task):
            finish_num+=1
            print('[{}]************已完成：{}/{}********************'.format(datetime.datetime.now(),finish_num,len(all_task)))
    print("程序结束%s" % datetime.datetime.now())


