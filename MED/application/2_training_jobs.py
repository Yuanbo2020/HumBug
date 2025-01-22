import os
from time import sleep
from multiprocessing import Pool
import multiprocessing as mp


# 这里的0是GPU id
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def job_all(model_name, repeat_time, batch, using_adamw, check_step, filepath, lr_rates, epochs):
    for repeat_no in range(repeat_time):
        for adamw in using_adamw:
            for lr_rate in lr_rates:
                if adamw:
                    command = 'python {0}   -repeat_no {1} -batch {2} -lr_rate {3} ' \
                              '   -epochs {4}   -check_step {5}  -model_name {6}  -adamw'.format(filepath,
                                                                                repeat_no,
                                                                                batch,
                                                                                lr_rate,
                                                                                epochs,
                                                                                check_step, model_name)
                else:
                    command = 'python {0}   -repeat_no {1} -batch {2} -lr_rate {3} ' \
                              '   -epochs {4}   -check_step {5}  -model_name {6} '.format(filepath,
                                                                                repeat_no,
                                                                                batch,
                                                                                lr_rate,
                                                                                epochs,
                                                                                check_step, model_name)





                print(command)
                if os.system(command):  # 成功返回0
                    print('\nFailed: ', command, '\n')
                    sleep(1)



def main():

    # import pynvml
    # from time import sleep
    # import time
    # pynvml.nvmlInit()
    #
    # # 这里的0是GPU id
    # handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #
    # gb = meminfo.used / 1024 / 1024 / 1024
    #
    # print('gb: ', gb)
    #
    # while gb > 0.01:
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    #     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #     gb = meminfo.used / 1024 / 1024 / 1024
    #     print('gb: ', gb, time.time())
    #     sleep(600)

    model_name = 'MTRCNN'
    # model_name = 'VGGishDropoutFeatB'
    # model_name = 'PANN'
    # model_name = 'MobileNetV2'
    # model_name = 'YAMNet'
    # model_name = 'CNN_Transformer'

    filepath = os.path.join(os.getcwd(), '2_training_main.py')

    repeat_time = 5
    using_adamw = [False]
    epochs = 100
    check_step = 1
    batchs = [32, 64]
    batch = batchs[0]
    lr_rates = [0.0005]

    if model_name == 'VGGishDropoutFeatB':
        lr_rates = [0.0003]

    cpu_num = mp.cpu_count()
    print('cpu_num: ', cpu_num)

    cpu_num = 4
    pool = Pool(cpu_num)
    # pool=Pool(最大的进程数)
    # 然后添加多个需要执行的进程，可以大于上面设置的最大进程数，会自动按照特定的队列顺序执行

    if cpu_num == 1:
        pool.apply_async(func=job_all, args=(model_name, repeat_time, batch, using_adamw, check_step, filepath, lr_rates, epochs))
    else:
        for i in range(cpu_num * repeat_time * len(lr_rates)):
            pool.apply_async(func=job_all, args=(model_name, repeat_time, batch, using_adamw, check_step, filepath, lr_rates, epochs))

    pool.close()
    pool.join()
    # join(): 等待工作进程结束。调用 join() 前必须先调用 close() 或者 terminate() 。


if __name__ == '__main__':
    main()

