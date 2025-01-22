import sys, os, argparse, torch, pickle
from framework.data_generator import DataGenerator_Yuanbo_inference_others
from framework.models import *
from framework.processing import training_process
from framework.utilities import create_folder
import framework.config as config
from framework.evaluate import get_results_only_one_class_label, evaluate_model
import numpy as np
from sklearn.metrics import accuracy_score

# gpu_id = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    model_name = 'MTRCNN'
    # model_name = 'VGGishDropoutFeatB'
    # model_name = 'PANN'
    # model_name = 'MobileNetV2'
    # model_name = 'YAMNet'
    # model_name = 'CNN_Transformer'

    repeat_time = 5
    using_adamw = [False]
    epochs = 100
    batchs = [32, 64]
    batch_size = batchs[0]
    lr_rates = [0.0005]

    if model_name == 'VGGishDropoutFeatB':
        lr_rates = [0.0003]

    lr_init = lr_rates[0]

    cuda = 1
    model_type_list = ['best_val', 'final_model']
    MC_dropout = False
    re_test = False

    replot = True
    nolabel = True

    # inference_audio = '1_audio_aedes_albopictus'
    inference_audio = '2_Audio_Stella_Nucr13_Aug18_midframe'

    filename = os.path.basename(__file__).split('.py')[0]
    inference_dir = os.path.join(os.getcwd(), filename)
    create_folder(inference_dir)

    for repeat_no in range(repeat_time):
        for adamw in using_adamw:
            for model_type in model_type_list:

                sys_name = model_name + '_' + str(repeat_no) + '_lr' + str(lr_init).replace('-', '') + '_e' + str(epochs) + '_b' + str(batch_size)
                workspace = os.path.join(os.getcwd(), sys_name)

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print(f'Evaluating on {device}')

                current_model_dir = os.path.join(inference_dir, sys_name)
                create_folder(current_model_dir)

                filename = inference_audio + '_' + model_name + '_' + model_type

                if MC_dropout:
                    filename += '_MC_dropout'
                preditions_file = os.path.join(current_model_dir, filename + '.pickle')

                if re_test or not os.path.exists(preditions_file):
                    using_model = eval(model_name)

                    if model_name == 'VGGishDropoutFeatB':
                        model = using_model(preprocess=False, dropout=0.2)
                    elif model_name == 'MobileNetV2' or model_name == 'YAMNet' or model_name == 'CNN_Transformer':
                        model = using_model(class_num=1, dropout=0.2, MC_dropout=MC_dropout)
                    else:
                        model = using_model(class_num=1, dropout=0.2, MC_dropout=MC_dropout, batchnormal=True)

                    model_path = os.path.join(workspace, 'model', model_type + '.pth')
                    pretrained_model = torch.load(model_path, map_location=device)
                    model.load_state_dict(pretrained_model, strict=True)

                    if config.cuda and torch.cuda.is_available():
                        model.cuda()

                    all_batch_y = []
                    all_batch_names = []
                    all_pred_y = []
                    all_batch_x = []
                    generator = DataGenerator_Yuanbo_inference_others(batch_size, inference_audio=inference_audio, nolabel=nolabel)
                    generate_func = generator.generate_testing_others_1s_clip_matrix(data_type='test', test_type=inference_audio)

                    for num, data in enumerate(generate_func):
                        (batch_x_cpu, batch_y, batch_name) = data
                        all_batch_x.append(batch_x_cpu)
                        batch_x = move_data_to_gpu(batch_x_cpu, cuda)

                        model.eval()

                        with torch.no_grad():
                            y_pred_sigmoid = model(batch_x)
                            all_batch_names.append(batch_name)
                            all_pred_y.append(y_pred_sigmoid.data.cpu().numpy())

                            if not nolabel:
                                all_batch_y.append(batch_y)

                    all_batch_x = np.concatenate(all_batch_x)
                    all_batch_names = np.concatenate(all_batch_names)
                    all_pred_y = np.concatenate(all_pred_y)

                    if not nolabel:
                        all_batch_y = np.concatenate(all_batch_y)

                    if not nolabel:
                        print('y_preds_all: ', all_pred_y.shape, all_batch_y.shape)  # (30, 3364, 2) (30, 3364, 1)

                        results = {'all_pred_y': all_pred_y, 'all_batch_y': all_batch_y, 'all_batch_names': all_batch_names,
                                   'all_batch_x': all_batch_x}
                    else:
                        results = {'all_pred_y': all_pred_y,
                                   'all_batch_names': all_batch_names,
                                   'all_batch_x': all_batch_x}
                    with open(preditions_file, 'wb') as f:
                        pickle.dump(results, f)
                else:
                    print('loading: ', preditions_file)
                    with open(preditions_file, 'rb') as input_file:
                        results = pickle.load(input_file)
                        all_pred_y = results['all_pred_y']
                        all_batch_x = results['all_batch_x']
                        all_batch_names = results['all_batch_names']

                        if not nolabel:
                            all_batch_y = results['all_batch_y']

                # print(all_pred_y.shape, all_batch_y.shape, all_batch_names.shape, all_batch_x.shape)
                # # (665, 1) (665, 1) (665,) (665, 1, 100, 64)

                if not nolabel:
                    test_acc = accuracy_score(all_batch_y, (all_pred_y > 0.5).astype(float))
                    print('n_test_acc:', test_acc)

                name_set = set(all_batch_names)
                # print(name_set)

                for each_name in name_set:
                    index = np.where(all_batch_names == each_name)[0]
                    # print(index)
                    mel_data = all_batch_x[index][:, 0].reshape(-1, config.mel_bins)
                    print('all_pred_y[index]: ', all_pred_y[index])
                    # print('all_pred_y[index]: ', all_pred_y.shape)
                    pred_data = all_pred_y[index].repeat(100)

                    plot_dir = os.path.join(current_model_dir, filename + '_plot' )
                    create_folder(plot_dir)
                    png_file = os.path.join(plot_dir, each_name.replace('.npy', '.png'))

                    if replot or not os.path.exists(png_file):
                        print(each_name, len(index), mel_data.shape, pred_data.shape)
                        # 120619_0065m.WAV.npy 55 (5500, 64) (55, 1)

                        clip_matrix_dir = os.path.join(current_model_dir, filename + '_prediction_1s_level')
                        create_folder(clip_matrix_dir)
                        file = os.path.join(clip_matrix_dir, each_name.replace('.npy', '_1s.txt'))
                        np.savetxt(file, all_pred_y[index])

                        clip_matrix_dir = os.path.join(current_model_dir, filename + '_prediction_10ms_level')
                        create_folder(clip_matrix_dir)
                        file = os.path.join(clip_matrix_dir, each_name.replace('.npy', '.txt'))
                        np.savetxt(file, pred_data)

                        import matplotlib.pyplot as plt
                        plt.close()

                        fig = plt.figure(figsize=(50, 10))
                        axs1 = plt.subplot(2, 1, 1)
                        axs1.set_xlim(xmin=0, xmax=mel_data.shape[0])
                        axs1.set_ylim(ymin=0, ymax=config.mel_bins)
                        axs1.matshow(mel_data.T, origin='lower', aspect='auto', cmap='jet')
                        # axs[0].xlim(0, len(mel_data))
                        # axs[0].ylim(0, config.mel_bins)

                        axs2 = plt.subplot(2, 1, 2)
                        axs2.set_xlim(xmin=0, xmax=len(pred_data))
                        axs2.set_ylim(ymin=0, ymax=1.1)
                        axs2.plot(pred_data)
                        plt.tight_layout()

                        plt.savefig(png_file)
                        # plt.show()
                        plt.close()



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















