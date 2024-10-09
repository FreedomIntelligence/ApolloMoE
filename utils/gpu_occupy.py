import subprocess
import time

def get_gpu_utilization():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8')
        gpu_utilization = [float(line.strip()) for line in output.splitlines()]
        return gpu_utilization[0]
    except subprocess.CalledProcessError:
        return 0

last_gpu_run_time = time.time()

while True:
    gpu_utilization = get_gpu_utilization()
    print(gpu_utilization)
    if gpu_utilization > 10:
        last_gpu_run_time = time.time()
    else:
        current_time = time.time()
        idle_time = current_time - last_gpu_run_time
        if idle_time > 5 * 60:
            print("GPU has been idle for more than 3 minutes. Exiting program.")
            break

    time.sleep(60)

print("Program ended.")