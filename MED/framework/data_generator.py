import pandas as pd
import numpy as np
import os
from framework.utilities import create_folder, calculate_scalar
from sklearn.model_selection import train_test_split
import time
import os
import skimage.util
import numpy as np
import framework.config as config
import pickle
import math
import collections
import pandas as pd
from sklearn.utils import shuffle



class count_the_audio_dataset(object):
    """
    1) 使用来自PANN的mel设置, 1 frame = 10 ms
    2) 由于不同片段长度差异实在过大，所以选用 1 秒的片段，在 1 秒级别进行，虽然这样还是避免不了 label_y 的不准确
    因为可能是20秒的片段，只有0~3秒有蚊子，如‘220523.wav’，最开始是一段说话声，但是标签是 mosquito
    """

    def __init__(self, batch_size, seed=42, training=True):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        load_time = time.time()

        csv_file = 'neurips_2021_zenodo_0_0_1.csv'
        df = pd.read_csv(os.path.join(config.metadata_space, csv_file))

        # To be kept: please do not edit the test set: these paths select test set A, test set B as described in the paper
        idx_test_A = np.logical_and(df['country'] == 'Tanzania', df['location_type'] == 'field')
        idx_test_B = np.logical_and(df['country'] == 'UK', df['location_type'] == 'culture')

        idx_train = np.logical_not(np.logical_or(idx_test_A, idx_test_B))

        df_test_A = df[idx_test_A]
        df_test_B = df[idx_test_B]

        df_train = df[idx_train]

        # Assertion to check that train does NOT appear in test:
        assert len(np.where(pd.concat([df_train, df_test_A,
                                       df_test_B]).duplicated())[
                       0]) == 0, 'Train dataframe contains overlap with Test A, Test B'

        FeatB_space = os.path.join(config.Data_space, 'Log_mel_YB_PANN')
        self.X_train, self.y_train, self.X_test_A, self.y_test_A, self.X_test_B, self.y_test_B = self.get_train_test_from_pickle(
            FeatB_space, df_train, df_test_A,
            df_test_B, training=training)

        print(self.X_test_A.shape, self.X_test_B.shape)
        # # (7533, 1, 100, 64) (3364, 1, 100, 64)



    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def get_train_test_from_pickle(self, FeatB_space, df_train, df_test_A, df_test_B, training=True):
        create_folder(FeatB_space)
        # print('FeatB_space: ', FeatB_space)

        feature_pool = []
        source_feature_dir = config.audio_space + "_PANN_mel_8kHz"
        for each in os.listdir(source_feature_dir):
            if each.endswith('.npy'):
                feature_pool.append(each.split('.npy')[0])

        pickle_name_train = 'log_mel_PANN_train_' + str(config.mel_bins) + '_win_' + str(config.win_size) \
                            + '_step_' + str(config.step_size) + '.pickle'
        print('pickle_name_train: ', pickle_name_train)

        print('Extracting training features...')
        X_train, y_train, skipped_files_train = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                                      data_df=df_train,
                                                                                      min_duration=config.min_duration)


        # total_frames:  10890991 * 0.01 s = 30.25 h
        # X_train:  7434 (112189, 64)
        # y_train:  7434 0
        # validation: 0.2 = 6.5 h

        print('X_train: ', len(X_train), X_train[0].shape)
        print('y_train: ', len(y_train), y_train[0])
        # X_train:  (302037, 1, 30, 128)
        # y_train:  (302037,)
        # 相当于把 feat:  (128, 17530) (mel bins, frames) 17530 帧的音频，切分成 30 帧的小段，然后判断这30帧是否有目标声音，是一个二分类
        # Frame duration in ms  2048/4/8000=0.064s=64ms
        # 30 * 64 = 192 ms = 0.192s
        # 2025-1-13, PANN mel
        # X_train:  (207066, 1, 100, 64)
        # y_train:  (207066,)

        # print('y_train dtype: ', y_train.dtype)  # y_train dtype:  int32
        # print('y_train: ', y_train)  # y_train:  [0 0 0 ... 1 1 1]

        pickle_name_test = 'log_mel_PANN_test_' + str(config.mel_bins) + '_win_' + str(config.win_size) \
                            + '_step_' + str(config.step_size) + '.pickle'
        print('pickle_name_test: ', pickle_name_test)

        print('Extracting test features...')

        X_test_A, y_test_A, skipped_files_test_A = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                          data_df=df_test_A,
                                                                          min_duration=config.min_duration)
        X_test_B, y_test_B, skipped_files_test_B = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                             data_df=df_test_B,
                                                                             min_duration=config.min_duration)

        # total_frames:  810472 * 0.01 s = 2.25 h
        # Completed 100 of 198

        # total_frames:  344477 * 0.01 s = 0.95 h
        # X_test_A:  (256, 64)
        # y_test_A:  1
        # X_test_B:  (5377, 64)
        # y_test_B:  0

        print('X_test_A: ', X_test_A[0].shape)
        print('y_test_A: ', y_test_A[0])
        # X_test_A:  (3782, 1, 30, 128)
        # y_test_A:  (3782,)

        print('X_test_B: ', X_test_B[0].shape)
        print('y_test_B: ', y_test_B[0])
        # X_test_B:  (1700, 1, 30, 128)
        # y_test_B:  (1700,)
        # 30 * 64 = 192 ms = 0.192s， 1700 个 0.192 s

        jjjjj

        if training:
            return X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B
        else:
            return None, None, X_test_A, y_test_A, X_test_B, y_test_B

    def get_feat_from_pickle(self, source_feature_dir, feature_pool, data_df, min_duration):
        """
        Returns features extracted with Librosa. A list of features, with the number
                of items equal to the number of input recordings
        :param data_df:
        :param data_dir:
        :param rate:
        :param min_duration:
        :param mel_bins:
        :return:
        """

        total_frames = 0
        X = []
        y = []
        idx = 0
        skipped_files = []

        for row_idx_series in data_df.iterrows():
            idx += 1
            if idx % 100 == 0:
                print('Completed', idx, 'of', len(data_df))
            row = row_idx_series[1]
            # print(row)
            # # id                                       199980
            # # length                                  1121.88
            # # name                             background.wav
            # # sample_rate                                8000
            # # record_datetime                  08-09-16 08:00
            # # sound_type                           background
            # # species                                     NaN
            # # gender                                      NaN
            # # fed                                         NaN
            # # plurality                                   NaN
            # # age                                         NaN
            # # method                                      NaN
            # # mic_type                                  phone
            # # device_type                       Alcatel 4009X
            # # country                                     USA
            # # district                                Georgia
            # # province                                Atlanta
            # # place              CDC insect cultures, Atlanta
            # # location_type                           culture
            # # Name: 0, dtype: object
            # print(label_duration)  # 1121.88

            _, file_format = os.path.splitext(row['name'])
            filename = (str(row['id']) + file_format).split('.wav')[0]

            if filename in feature_pool:
                file_path = os.path.join(source_feature_dir, filename + '.npy')
                feat = np.load(file_path)
                # print('feat: ', feat.shape)  # feat:  (70, 64)

                if len(feat) < min_duration:
                    diff = np.zeros((min_duration-len(feat), config.mel_bins))
                    feat = np.concatenate([feat, diff], axis=0)
                    # print('feat: ', feat.shape)  # feat:  (100, 64)

                X.append(feat)

                total_frames+=len(feat)
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                elif row['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                skipped_files.append(filename)

        # print('skipped_files: ', skipped_files)
        print('skipped_files len: ', len(skipped_files))
        # skipped_files len:  500
        print('total_frames: ', total_frames)
        return X, y, skipped_files




class DataGenerator_Yuanbo(object):
    """
    1) 使用来自PANN的mel设置, 1 frame = 10 ms
    2) 由于不同片段长度差异实在过大，所以选用 1 秒的片段，在 1 秒级别进行，虽然这样还是避免不了 label_y 的不准确
    因为可能是20秒的片段，只有0~3秒有蚊子，如‘220523.wav’，最开始是一段说话声，但是标签是 mosquito
    """

    def __init__(self, batch_size, seed=42, training=True):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        load_time = time.time()

        csv_file = 'neurips_2021_zenodo_0_0_1.csv'
        df = pd.read_csv(os.path.join(config.metadata_space, csv_file))

        # To be kept: please do not edit the test set: these paths select test set A, test set B as described in the paper
        idx_test_A = np.logical_and(df['country'] == 'Tanzania', df['location_type'] == 'field')
        idx_test_B = np.logical_and(df['country'] == 'UK', df['location_type'] == 'culture')
        # print(idx_test_A)
        # # 0       False
        # # 1       False
        # # 2       False
        # # 3       False
        # # 4       False
        # #         ...
        # # 9290    False
        # # 9291    False
        # # 9292    False
        # # 9293    False
        # # 9294    False
        # # Length: 9295, dtype: bool
        # print(idx_test_B)
        # # 0       False
        # # 1       False
        # # 2       False
        # # 3       False
        # # 4       False
        # #         ...
        # # 9290    False
        # # 9291    False
        # # 9292    False
        # # 9293    False
        # # 9294    False
        # # Length: 9295, dtype: bool

        idx_train = np.logical_not(np.logical_or(idx_test_A, idx_test_B))

        df_test_A = df[idx_test_A]
        df_test_B = df[idx_test_B]

        df_train = df[idx_train]

        # Assertion to check that train does NOT appear in test:
        assert len(np.where(pd.concat([df_train, df_test_A,
                                       df_test_B]).duplicated())[
                       0]) == 0, 'Train dataframe contains overlap with Test A, Test B'

        FeatB_space = os.path.join(config.Data_space, 'Log_mel_YB_PANN')
        self.X_train, self.y_train, self.X_test_A, self.y_test_A, self.X_test_B, self.y_test_B = self.get_train_test_from_pickle(
            FeatB_space, df_train, df_test_A,
            df_test_B, training=training)

        print(self.X_test_A.shape, self.X_test_B.shape)
        # # (7533, 1, 100, 64) (3364, 1, 100, 64)


        if training:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2,
                                                                                  random_state=42)

            print('X_train: ', self.X_train.shape)
            print('y_train: ', self.y_train.shape)
            # X_train:  (241629, 1, 30, 128)
            # y_train:  (241629,)
            # after
            # X_train:  (207066, 1, 100, 64)
            # y_train:  (207066,)

            print('X_val: ', self.X_val.shape)
            print('y_val: ', self.y_val.shape)
            # X_val:  (60408, 1, 30, 128)
            # y_val:  (60408,)
            # pann mel
            # X_val:  (41414, 1, 100, 64)
            # y_val:  (41414,)

            self.y_train = self.y_train.astype(np.float16)
            self.y_val = self.y_val.astype(np.float16)

        self.y_test_A = self.y_test_A.astype(np.float16)
        self.y_test_B = self.y_test_B.astype(np.float16)

        ################################################################################
        normalization_mel_file = os.path.join(FeatB_space, 'norm_mel.pickle')

        if not os.path.exists(normalization_mel_file):
            norm_pickle = {}
            print(self.X_train.shape, self.X_train[:, 0].shape)
            # (165652, 1, 100, 64) (165652, 100, 64)
            (self.mean_mel, self.std_mel) = calculate_scalar(self.X_train[:, 0])  # (165652, 1, 100, 64)
            norm_pickle['mean'] = self.mean_mel
            norm_pickle['std'] = self.std_mel
            self.save_pickle(norm_pickle, normalization_mel_file)

        else:
            print('using: ', normalization_mel_file)
            norm_pickle = self.load_pickle(normalization_mel_file)
            self.mean_mel = norm_pickle['mean']
            self.std_mel = norm_pickle['std']

        print(self.mean_mel)
        print(self.std_mel)
        # [-39.83863551 -43.46023656 -46.2777389  -50.5651904  -52.67609668
        #  -53.3281098  -53.81438447 -54.45044573 -54.97509048 -55.21784194
        #  -55.21666574 -55.97964111 -56.29681423 -56.43277034 -56.58488528
        #  -57.72267882 -59.57005133 -60.34445583 -60.31653225 -60.47045868
        #  -60.49732187 -60.72640398 -60.53009346 -60.51363672 -61.10558249
        #  -61.62595803 -63.14982927 -64.24964721 -64.73626253 -64.78599689
        #  -65.00343122 -66.14722445 -65.25158627 -66.17788285 -65.31174353
        #  -65.89234272 -65.59095784 -65.56977594 -65.89925691 -65.93424183
        #  -66.0118296  -66.0386418  -66.26122736 -66.70050188 -67.0295137
        #  -67.17761048 -67.56658001 -67.27762187 -67.75135913 -67.89415737
        #  -68.18929324 -68.47497696 -68.67873371 -68.99200627 -69.14608002
        #  -69.26571    -69.22440335 -69.21112846 -69.15825938 -69.16517422
        #  -69.30632383 -69.96022701 -72.73592342 -81.14826649]
        # [ 8.92037295 10.48276105 11.61662207 13.59732223 14.57550992 14.36251991
        #  13.71137033 13.02019505 12.53532825 11.97920706 11.70065432 12.00973609
        #  11.95536756 12.39475593 13.12850887 13.20097898 12.72378416 12.21179829
        #  11.72611651 11.29967956 11.09609857 11.26992297 11.56054049 11.58587109
        #  11.64393227 11.84215407 12.30185557 12.41580732 12.20131768 11.94167401
        #  11.85023619 11.8261591  11.44015834 11.29131607 11.08146564 10.96594261
        #  10.80123672 10.66298036 10.6554342  10.6700453  10.67404322 10.69556526
        #  10.66910069 10.77737902 10.88553395 10.9238569  10.79813069 10.59901917
        #  10.66271638 10.78767624 10.87505722 10.91654661 10.98118523 11.12754979
        #  11.20703754 11.14407169 11.00775173 10.88947608 10.75813649 10.65144918
        #  10.61683125 10.69334199 11.38074083 14.16355425]
        # norm:  (64,) (64,)
        print("norm: ", self.mean_mel.shape, self.std_mel.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def get_train_test_from_pickle(self, FeatB_space, df_train, df_test_A, df_test_B, training=True):
        create_folder(FeatB_space)
        # print('FeatB_space: ', FeatB_space)

        feature_pool = []
        source_feature_dir = config.audio_space + "_PANN_mel_8kHz"
        for each in os.listdir(source_feature_dir):
            if each.endswith('.npy'):
                feature_pool.append(each.split('.npy')[0])

        if training:
            pickle_name_train = 'log_mel_PANN_train_' + str(config.mel_bins) + '_win_' + str(config.win_size) \
                                + '_step_' + str(config.step_size) + '.pickle'
            print('pickle_name_train: ', pickle_name_train)

            if not os.path.isfile(os.path.join(FeatB_space, pickle_name_train)):
                print('Extracting training features...')
                X_train, y_train, skipped_files_train = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                                              data_df=df_train,
                                                                                              min_duration=config.min_duration)
                X_train, y_train = self.reshape_feat(X_train, y_train, config.win_size, config.step_size)

                log_mel_feat_train = {'X_train': X_train, 'y_train': y_train, 'skipped_files_train': skipped_files_train}

                with open(os.path.join(FeatB_space, pickle_name_train), 'wb') as f:
                    pickle.dump(log_mel_feat_train, f, protocol=4)
                    print('Saved features to:', os.path.join(FeatB_space, pickle_name_train))
            else:
                print('Loading training features found at:', os.path.join(FeatB_space, pickle_name_train))
                with open(os.path.join(FeatB_space, pickle_name_train), 'rb') as input_file:
                    log_mel_feat = pickle.load(input_file)
                    # print('log_mel_feat: ', log_mel_feat.keys())
                    X_train = log_mel_feat['X_train']
                    y_train = log_mel_feat['y_train']
            # print('X_train: ', X_train.shape)
            # print('y_train: ', y_train.shape)
            # X_train:  (302037, 1, 30, 128)
            # y_train:  (302037,)
            # 相当于把 feat:  (128, 17530) (mel bins, frames) 17530 帧的音频，切分成 30 帧的小段，然后判断这30帧是否有目标声音，是一个二分类
            # Frame duration in ms  2048/4/8000=0.064s=64ms
            # 30 * 64 = 192 ms = 0.192s
            # 2025-1-13, PANN mel
            # X_train:  (207066, 1, 100, 64)
            # y_train:  (207066,)

            # print('y_train dtype: ', y_train.dtype)  # y_train dtype:  int32
            # print('y_train: ', y_train)  # y_train:  [0 0 0 ... 1 1 1]

        pickle_name_test = 'log_mel_PANN_test_' + str(config.mel_bins) + '_win_' + str(config.win_size) \
                            + '_step_' + str(config.step_size) + '.pickle'
        print('pickle_name_test: ', pickle_name_test)

        if not os.path.isfile(os.path.join(FeatB_space, pickle_name_test)):
            print('Extracting test features...')

            X_test_A, y_test_A, skipped_files_test_A = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                              data_df=df_test_A,
                                                                              min_duration=config.min_duration)
            X_test_B, y_test_B, skipped_files_test_B = self.get_feat_from_pickle(source_feature_dir, feature_pool,
                                                                                 data_df=df_test_B,
                                                                                 min_duration=config.min_duration)
            X_test_A, y_test_A = self.reshape_feat(X_test_A, y_test_A, config.win_size, config.win_size)
            # Test should be strided with step = window.
            X_test_B, y_test_B = self.reshape_feat(X_test_B, y_test_B, config.win_size, config.win_size)

            log_mel_feat_test = {'X_test_A': X_test_A, 'X_test_B': X_test_B, 'y_test_A': y_test_A, 'y_test_B': y_test_B}
            # skipped_files len:  4

            with open(os.path.join(FeatB_space, pickle_name_test), 'wb') as f:
                pickle.dump(log_mel_feat_test, f, protocol=4)
                print('Saved features to:', os.path.join(FeatB_space, pickle_name_test))
        else:
            print('Loading test features found at:', os.path.join(FeatB_space, pickle_name_test))
            with open(os.path.join(FeatB_space, pickle_name_test), 'rb') as input_file:
                log_mel_feat = pickle.load(input_file)

                X_test_A = log_mel_feat['X_test_A']
                y_test_A = log_mel_feat['y_test_A']
                X_test_B = log_mel_feat['X_test_B']
                y_test_B = log_mel_feat['y_test_B']

        # print('X_test_A: ', X_test_A.shape)
        # print('y_test_A: ', y_test_A.shape)
        # # X_test_A:  (3782, 1, 30, 128)
        # # y_test_A:  (3782,)
        #
        # print('X_test_B: ', X_test_B.shape)
        # print('y_test_B: ', y_test_B.shape)
        # # X_test_B:  (1700, 1, 30, 128)
        # # y_test_B:  (1700,)
        # # 30 * 64 = 192 ms = 0.192s， 1700 个 0.192 s

        if training:
            return X_train, y_train, X_test_A, y_test_A, X_test_B, y_test_B
        else:
            return None, None, X_test_A, y_test_A, X_test_B, y_test_B

    def get_feat_from_pickle(self, source_feature_dir, feature_pool, data_df, min_duration):
        """
        Returns features extracted with Librosa. A list of features, with the number
                of items equal to the number of input recordings
        :param data_df:
        :param data_dir:
        :param rate:
        :param min_duration:
        :param mel_bins:
        :return:
        """

        X = []
        y = []
        idx = 0
        skipped_files = []

        for row_idx_series in data_df.iterrows():
            idx += 1
            if idx % 100 == 0:
                print('Completed', idx, 'of', len(data_df))
            row = row_idx_series[1]
            # print(row)
            # # id                                       199980
            # # length                                  1121.88
            # # name                             background.wav
            # # sample_rate                                8000
            # # record_datetime                  08-09-16 08:00
            # # sound_type                           background
            # # species                                     NaN
            # # gender                                      NaN
            # # fed                                         NaN
            # # plurality                                   NaN
            # # age                                         NaN
            # # method                                      NaN
            # # mic_type                                  phone
            # # device_type                       Alcatel 4009X
            # # country                                     USA
            # # district                                Georgia
            # # province                                Atlanta
            # # place              CDC insect cultures, Atlanta
            # # location_type                           culture
            # # Name: 0, dtype: object
            # print(label_duration)  # 1121.88

            _, file_format = os.path.splitext(row['name'])
            filename = (str(row['id']) + file_format).split('.wav')[0]

            if filename in feature_pool:
                file_path = os.path.join(source_feature_dir, filename + '.npy')
                feat = np.load(file_path)
                # print('feat: ', feat.shape)  # feat:  (70, 64)

                if len(feat) < min_duration:
                    diff = np.zeros((min_duration-len(feat), config.mel_bins))
                    feat = np.concatenate([feat, diff], axis=0)
                    # print('feat: ', feat.shape)  # feat:  (100, 64)

                X.append(feat)
                if row['sound_type'] == 'mosquito':
                    y.append(1)
                elif row['sound_type']:  # Condition to check we are not adding empty (or unexpected) labels as 0
                    y.append(0)
            else:
                skipped_files.append(filename)

        # print('skipped_files: ', skipped_files)
        print('skipped_files len: ', len(skipped_files))
        # skipped_files len:  500
        return X, y, skipped_files

    def reshape_feat(self, feats, labels, win_size, step_size):
        """
        X_train, y_train, config.win_size, config.step_size
        前两个都是列表，每个元素的长度不一
        :param feats:
        :param labels:
        :param win_size: win_size = 30
        :param step_size: step_size = 5
        :return:
        # feat =np.zeros((128, 100))
        # win_size = 30
        # step_size = 5
        # feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, np.shape(feat)[0]), step=step_size)
        # # print('feats_windowed: ', feats_windowed.shape)  # feats_windowed:  (15, 1, 30, 128)
        # # 100-30=70/5=14 + 1 (来自30)=15
        """
        '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is
        given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
        Can code to be a function of time and hop length instead in future.'''

        feats_windowed_array = []
        labels_windowed_array = []
        for idx, feat in enumerate(feats):
            # print('feat: ', feat.shape)  # feat:  (112189, 64)
            feat = feat.T
            if np.shape(feat)[1] < win_size:
                print('Length of recording shorter than supplied window size.')
                pass
            else:
                # print('feat: ', feat.shape)    # feat:  (64, 112189)
                feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, config.mel_bins), step=step_size)
                # print('feats_windowed: ', feats_windowed.shape)  # (2242, 1, 100, 64)
                # # 112189 - 100 = 112089 / 50 = 2241.78 = 2241 + 1 = 2242
                labels_windowed = np.full(len(feats_windowed), labels[idx])
                feats_windowed_array.append(feats_windowed)
                labels_windowed_array.append(labels_windowed)
        return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)

    def generate_training(self):
        audios_num = len(self.y_train)

        audio_indexes = [i for i in range(audios_num)]
        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.X_train[batch_audio_indexes]
            batch_x = self.transform(batch_x[:, 0], self.mean_mel, self.std_mel)[:, None]
            batch_y = self.y_train[batch_audio_indexes]

            # print(batch_x.shape, batch_y.shape)
            # # (32, 1, 100, 64) (32,)

            yield batch_x, batch_y[:, None]

    def generate_validate(self, data_type, max_iteration=None):
        # load
        # ------------------ validation --------------------------------------------------------------------------------

        audios_num = len(self.y_val)
        audio_indexes = [i for i in range(audios_num)]
        self.validate_random_state.shuffle(audio_indexes)
        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.X_val[batch_audio_indexes]
            batch_x = self.transform(batch_x[:, 0], self.mean_mel, self.std_mel)[:, None]
            batch_y = self.y_val[batch_audio_indexes]
            yield batch_x, batch_y[:, None]

    def generate_testing(self, data_type, test_type, max_iteration=None):
        # # 释放
        # try:
        #     if self.using_mel:
        #         self.val_all_feature_data
        # except NameError:
        #     var_exists = False
        # else:
        #     var_exists = True
        # print('\n\nvar_exists: ', var_exists)
        #
        # if delete_val and var_exists:
        #     if self.using_mel:
        #         del self.val_all_feature_data
        #         del self.val_x
        #     if self.using_loudness:
        #         del self.val_all_feature_data_loudness
        #         del self.val_x_loudness
        #     gc.collect()
        #     torch.cuda.empty_cache()

        if test_type == 'test_A':
            Test_data = self.X_test_A
            Test_label = self.y_test_A

        if test_type == 'test_B':
            Test_data = self.X_test_B
            Test_label = self.y_test_B

        audios_num = len(Test_data)

        audio_indexes = [i for i in range(audios_num)]
        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break
            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = Test_data[batch_audio_indexes]
            batch_x = self.transform(batch_x[:, 0], self.mean_mel, self.std_mel)[:, None]
            batch_y = Test_label[batch_audio_indexes]
            # print('batch_y: ', batch_y)
            yield batch_x, batch_y[:, None]


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return (x - mean) / std




class DataGenerator_Yuanbo_inference_others(object):
    """
    1) 使用来自PANN的mel设置, 1 frame = 10 ms
    2) 由于不同片段长度差异实在过大，所以选用 1 秒的片段，在 1 秒级别进行，虽然这样还是避免不了 label_y 的不准确
    因为可能是20秒的片段，只有0~3秒有蚊子，如‘220523.wav’，最开始是一段说话声，但是标签是 mosquito
    """

    def __init__(self, batch_size, seed=42, inference_audio=None, nolabel=False):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.test_random_state = np.random.RandomState(0)

        load_time = time.time()

        feature_dir = os.path.join(config.Data_space, inference_audio) + "_PANN_mel_8kHz"

        X = []

        self.nolabel = nolabel
        if not nolabel:
            y = []

        audio_names = []
        for each in os.listdir(feature_dir):
            each_path = os.path.join(feature_dir, each)
            data = np.load(each_path)
            # print(data.shape)  # (6377, 64)
            X.append(data)

            if not nolabel:
                if 'aedes_albopictus' in feature_dir:
                    y.append(1)
            audio_names.append(each)

        self.x_clips = X

        if not nolabel:
            self.X_test_1s_clip_matrix, self.y_test_1s_clip_matrix, self.audio_names_1s_clip_matrix = self.reshape_feat_audio_names(X, y, audio_names, config.win_size, config.win_size)
            print(self.X_test_1s_clip_matrix.shape, self.y_test_1s_clip_matrix.shape)
            # (665, 1, 100, 64) (665,)

            self.y_test_1s_clip_matrix = self.y_test_1s_clip_matrix.astype(np.float16)
        else:
            self.X_test_1s_clip_matrix, self.audio_names_1s_clip_matrix = self.reshape_feat(
                X, audio_names,  config.win_size, config.win_size)

        ################################################################################
        FeatB_space = os.path.join(config.Data_space, 'Log_mel_YB_PANN')
        normalization_mel_file = os.path.join(FeatB_space, 'norm_mel.pickle')

        print('using: ', normalization_mel_file)
        norm_pickle = self.load_pickle(normalization_mel_file)
        self.mean_mel = norm_pickle['mean']
        self.std_mel = norm_pickle['std']

        print(self.mean_mel)
        print(self.std_mel)
        # [-39.83863551 -43.46023656 -46.2777389  -50.5651904  -52.67609668
        #  -53.3281098  -53.81438447 -54.45044573 -54.97509048 -55.21784194
        #  -55.21666574 -55.97964111 -56.29681423 -56.43277034 -56.58488528
        #  -57.72267882 -59.57005133 -60.34445583 -60.31653225 -60.47045868
        #  -60.49732187 -60.72640398 -60.53009346 -60.51363672 -61.10558249
        #  -61.62595803 -63.14982927 -64.24964721 -64.73626253 -64.78599689
        #  -65.00343122 -66.14722445 -65.25158627 -66.17788285 -65.31174353
        #  -65.89234272 -65.59095784 -65.56977594 -65.89925691 -65.93424183
        #  -66.0118296  -66.0386418  -66.26122736 -66.70050188 -67.0295137
        #  -67.17761048 -67.56658001 -67.27762187 -67.75135913 -67.89415737
        #  -68.18929324 -68.47497696 -68.67873371 -68.99200627 -69.14608002
        #  -69.26571    -69.22440335 -69.21112846 -69.15825938 -69.16517422
        #  -69.30632383 -69.96022701 -72.73592342 -81.14826649]
        # [ 8.92037295 10.48276105 11.61662207 13.59732223 14.57550992 14.36251991
        #  13.71137033 13.02019505 12.53532825 11.97920706 11.70065432 12.00973609
        #  11.95536756 12.39475593 13.12850887 13.20097898 12.72378416 12.21179829
        #  11.72611651 11.29967956 11.09609857 11.26992297 11.56054049 11.58587109
        #  11.64393227 11.84215407 12.30185557 12.41580732 12.20131768 11.94167401
        #  11.85023619 11.8261591  11.44015834 11.29131607 11.08146564 10.96594261
        #  10.80123672 10.66298036 10.6554342  10.6700453  10.67404322 10.69556526
        #  10.66910069 10.77737902 10.88553395 10.9238569  10.79813069 10.59901917
        #  10.66271638 10.78767624 10.87505722 10.91654661 10.98118523 11.12754979
        #  11.20703754 11.14407169 11.00775173 10.88947608 10.75813649 10.65144918
        #  10.61683125 10.69334199 11.38074083 14.16355425]
        # norm:  (64,) (64,)
        print("norm: ", self.mean_mel.shape, self.std_mel.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def reshape_feat_audio_names(self, feats, labels, audio_names, win_size, step_size):
        """
        X_train, y_train, config.win_size, config.step_size
        前两个都是列表，每个元素的长度不一
        :param feats:
        :param labels:
        :param win_size: win_size = 30
        :param step_size: step_size = 5
        :return:
        # feat =np.zeros((128, 100))
        # win_size = 30
        # step_size = 5
        # feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, np.shape(feat)[0]), step=step_size)
        # # print('feats_windowed: ', feats_windowed.shape)  # feats_windowed:  (15, 1, 30, 128)
        # # 100-30=70/5=14 + 1 (来自30)=15
        """
        '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is
        given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
        Can code to be a function of time and hop length instead in future.'''

        feats_windowed_array = []
        labels_windowed_array = []
        audio_names_windowed_array = []
        for idx, feat in enumerate(feats):
            # print('feat: ', feat.shape)  # feat:  (112189, 64)
            feat = feat.T
            if np.shape(feat)[1] < win_size:
                print('Length of recording shorter than supplied window size.')
                pass
            else:
                # print('feat: ', feat.shape)    # feat:  (64, 112189)
                feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, config.mel_bins), step=step_size)
                # print('feats_windowed: ', feats_windowed.shape)  # (2242, 1, 100, 64)
                # # 112189 - 100 = 112089 / 50 = 2241.78 = 2241 + 1 = 2242
                labels_windowed = np.full(len(feats_windowed), labels[idx])

                audio_names_windowed = np.full(len(feats_windowed), audio_names[idx])
                audio_names_windowed_array.append(audio_names_windowed)

                feats_windowed_array.append(feats_windowed)
                labels_windowed_array.append(labels_windowed)
        return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array), np.hstack(audio_names_windowed_array)

    def reshape_feat(self, feats, labels, win_size, step_size):
        """
        X_train, y_train, config.win_size, config.step_size
        前两个都是列表，每个元素的长度不一
        :param feats:
        :param labels:
        :param win_size: win_size = 30
        :param step_size: step_size = 5
        :return:
        # feat =np.zeros((128, 100))
        # win_size = 30
        # step_size = 5
        # feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, np.shape(feat)[0]), step=step_size)
        # # print('feats_windowed: ', feats_windowed.shape)  # feats_windowed:  (15, 1, 30, 128)
        # # 100-30=70/5=14 + 1 (来自30)=15
        """
        '''Reshaping features from get_feat to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is
        given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
        Can code to be a function of time and hop length instead in future.'''

        feats_windowed_array = []
        labels_windowed_array = []
        for idx, feat in enumerate(feats):
            # print('feat: ', feat.shape)  # feat:  (112189, 64)
            feat = feat.T
            if np.shape(feat)[1] < win_size:
                print('Length of recording shorter than supplied window size.')
                pass
            else:
                # print('feat: ', feat.shape)    # feat:  (64, 112189)
                feats_windowed = skimage.util.view_as_windows(feat.T, (win_size, config.mel_bins), step=step_size)
                # print('feats_windowed: ', feats_windowed.shape)  # (2242, 1, 100, 64)
                # # 112189 - 100 = 112089 / 50 = 2241.78 = 2241 + 1 = 2242
                labels_windowed = np.full(len(feats_windowed), labels[idx])
                feats_windowed_array.append(feats_windowed)
                labels_windowed_array.append(labels_windowed)
        return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)


    def generate_testing_others_1s_clip_matrix(self, data_type, test_type, max_iteration=None):
        # # 释放
        # try:
        #     if self.using_mel:
        #         self.val_all_feature_data
        # except NameError:
        #     var_exists = False
        # else:
        #     var_exists = True
        # print('\n\nvar_exists: ', var_exists)
        #
        # if delete_val and var_exists:
        #     if self.using_mel:
        #         del self.val_all_feature_data
        #         del self.val_x
        #     if self.using_loudness:
        #         del self.val_all_feature_data_loudness
        #         del self.val_x_loudness
        #     gc.collect()
        #     torch.cuda.empty_cache()

        Test_data = self.X_test_1s_clip_matrix

        if not self.nolabel:
            Test_label = self.y_test_1s_clip_matrix

        audios_num = len(Test_data)

        audio_indexes = [i for i in range(audios_num)]
        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break
            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = Test_data[batch_audio_indexes]
            batch_x = self.transform(batch_x[:, 0], self.mean_mel, self.std_mel)[:, None]

            if not self.nolabel:
                batch_y = Test_label[batch_audio_indexes]
                # print('batch_y: ', batch_y)

            name = self.audio_names_1s_clip_matrix[batch_audio_indexes]

            if not self.nolabel:
                yield batch_x, batch_y[:, None], name
            else:
                yield batch_x, None, name


    def generate_testing_others_clips(self, data_type, test_type, max_iteration=None):

        Test_data = self.X_test_1s_clip_matrix
        Test_label = self.y_test_1s_clip_matrix

        audios_num = len(Test_data)

        audio_indexes = [i for i in range(audios_num)]
        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0
        while True:
            if iteration == max_iteration:
                break
            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = Test_data[batch_audio_indexes]
            batch_x = self.transform(batch_x[:, 0], self.mean_mel, self.std_mel)[:, None]
            batch_y = Test_label[batch_audio_indexes]
            # print('batch_y: ', batch_y)

            name = self.audio_names_1s_clip_matrix[batch_audio_indexes]
            yield batch_x, batch_y[:, None]


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return (x - mean) / std



