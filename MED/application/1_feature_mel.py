import pickle
import sys, os, h5py, time, librosa, torch
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import os, time
import psutil
from time import sleep
from multiprocessing import Pool
import multiprocessing as mp


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def save_pickle(file, dict):
    with open(file, 'wb') as f:
        pickle.dump(dict, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def run_jobs():

    source_audio_dir = r'E:\Yuanbo\Code\26_MED_MSC\0_Dataset\audio'

    output_feature_dir = source_audio_dir + "_mel_8kHz"
    create_folder(output_feature_dir)

    mel_bins = 64
    sample_rate = 8000  # 16000
    fmax = int(sample_rate / 2)
    fmin = 50
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    # window_size = 512  # 16Khz
    # hop_size = 160  # 16Khz
    window_size = int(np.floor(1024 * (sample_rate / 32000)))
    hop_size = int(np.floor(320 * (sample_rate / 32000)))  # 10ms
    # print(window_size, hop_size)  # 256 80

    min_len = 0.5  # 最短0.5s, 500ms

    spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                        win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                                        freeze_parameters=True)

    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                        freeze_parameters=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    spectrogram_extractor.to(device)
    logmel_extractor.to(device)

    too_short_list = []
    for (file_n, audio_file_name) in enumerate(os.listdir(source_audio_dir)):
        audio_path = os.path.join(source_audio_dir, audio_file_name)

        output_feature_path = os.path.join(output_feature_dir,  audio_file_name.replace('.wav', '.npy'))

        if not os.path.exists(output_feature_path):

            original_sr = librosa.get_samplerate(audio_path)
            length = librosa.get_duration(filename=audio_path)
            print('Now: ', audio_path, ' length: ', length, ' original_sr: ', original_sr)  # (441000,) 44100

            if length >= min_len:
                audiodata, fs = librosa.core.load(audio_path, sr=sample_rate, mono=False)

                # 这里需要重新载入，因为采样率不一样
                # print('audiodata: ', audiodata.shape, fs)  # (159602,) 8000

                # 这里只对于每个音频提取特征，不规整到一块
                # if len(audiodata) < length * fs:
                #     diff = int(length * fs - len(audiodata))
                #     audiodata = np.concatenate([audiodata, audiodata[:diff]])
                # elif len(audiodata) > length * fs:
                #     audiodata = audiodata[:int(length * fs)]
                # print(audiodata.shape, fs)

                audiodata = np.array([audiodata, audiodata])  # 这里变为binaural，因为后面的需要
                # print(audiodata.shape)  # (2, 159602)
                # audio = pad_or_truncate(audiodata, clip_samples)

                spectrogram = spectrogram_extractor(move_data_to_device(audiodata, device))
                # print(spectrogram.shape)  # torch.Size([2, 1, 1996, 129])

                logmel = logmel_extractor(spectrogram)  # torch.Size([2, 1, 1001, 64])
                # print(logmel.shape)  # torch.Size([2, 1, 1996, 64])

                logmel = logmel[:, 0].data.cpu().numpy()
                print(file_n, audio_file_name, logmel.shape)  # 1 1000.wav (2, 1996, 64)

                # output_feature_path = os.path.join(output_feature_dir, audio_file_name.replace('.wav', '.pickle'))
                # # 998 KB
                # save_pickle(output_feature_path, logmel)
                # data = load_pickle(output_feature_path)
                # print(data.shape)  # (2, 1996, 64)
                #
                # output_feature_path = os.path.join(output_feature_dir, audio_file_name.replace('.wav', '.npy'))
                # # 998 KB
                # np.save(output_feature_path, logmel)
                # print(type(logmel), logmel.dtype)  # <class 'numpy.ndarray'> float32
                #
                # output_feature_path = os.path.join(output_feature_dir, audio_file_name.replace('.wav', '.npz'))
                # # 998 KB
                # np.savez(output_feature_path, logmel)
                # print(type(logmel), logmel.dtype)

                # 以上3中格式，float32都是998KB，所以直接采用npy吧
                # sub_dir_n 从 0 开始
                output_feature_path = os.path.join(output_feature_dir, audio_file_name.replace('.wav', '.npy'))
                # print(output_feature_path)
                np.save(output_feature_path, logmel[0])

            else:
                print('Too short')
                too_short_list.append(audio_file_name)

    print('final Too short list')
    print(too_short_list)
    print(len(too_short_list))
    filename = os.path.basename(__file__).split('.py')[0]
    file = os.path.join(os.getcwd(), filename + '_less_than_' + str(min_len).replace('.', '') + '.txt')
    with open(file, 'w') as f:
        for each in too_short_list:
            f.write(each + '\n')

    # too short: 505
    # 505+8791=9296, 和原始音频数目一致


def main(argv):
    run_jobs()

    # psutil.cpu_count()  # CPU逻辑数量
    # psutil.cpu_count(logical=False)  # CPU物理核心
    #
    # using = psutil.cpu_percent(interval=1, percpu=True)
    # # print(psutil.cpu_percent(interval=1, percpu=True))
    # print(np.mean(using))
    #
    # while (np.mean(using) > 35):
    #     time.sleep(10)
    #     using = psutil.cpu_percent(interval=1, percpu=True)
    #     # print(psutil.cpu_percent(interval=1, percpu=True))
    #     print(np.mean(using))
    #
    #
    # cpu_num = mp.cpu_count()
    # print('cpu_num: ', cpu_num)
    #
    # cpu_num = 5
    # pool = Pool(cpu_num)
    # # pool=Pool(最大的进程数)
    # # 然后添加多个需要执行的进程，可以大于上面设置的最大进程数，会自动按照特定的队列顺序执行
    #
    # if cpu_num == 1:
    #     pool.apply_async(func=run_jobs, args=())
    # else:
    #     for i in range(cpu_num*20):
    #         pool.apply_async(func=run_jobs, args=())
    #
    # pool.close()
    # pool.join()
    # #  join(): 等待工作进程结束。调用 join() 前必须先调用 close() 或者 terminate() 。


if __name__=="__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















