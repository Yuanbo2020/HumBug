import sys, os, argparse, torch, pickle
from framework.data_generator import DataGenerator_Yuanbo
from framework.models import VGGishDropoutFeatB, PANN, CNN_Transformer, move_data_to_gpu
from framework.processing import training_process
from framework.utilities import create_folder
import framework.config as config
from framework.evaluate import get_results, evaluate_model
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

    n_samples = 30
    cuda = 1
    test_type_list = ['test_A', 'test_B']
    model_type_list = ['best_val', 'final_model']
    MC_dropout = True
    re_test = False

    for repeat_no in range(repeat_time):
        for adamw in using_adamw:
            for model_type in model_type_list:

                sys_name = model_name + '_' + str(repeat_no) + '_lr' + str(lr_init).replace('-', '') + '_e' + str(epochs) + '_b' + str(batch_size)
                workspace = os.path.join(os.getcwd(), sys_name)

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print(f'Evaluating on {device}')

                for test_type in test_type_list:
                    inference_dir = os.path.join(workspace, 'inference_' + test_type)
                    create_folder(inference_dir)

                    filename = model_name + '_' + model_type + '_' + test_type

                    if MC_dropout:
                        filename += '_MC_dropout'
                    preditions_file = os.path.join(inference_dir, filename + '.pickle')

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

                        y_preds_all = []
                        for n in range(n_samples):

                            all_batch_y = []
                            all_pred_y = []
                            generator = DataGenerator_Yuanbo(batch_size, training=False)
                            generate_func = generator.generate_testing(data_type='test', test_type=test_type)

                            for num, data in enumerate(generate_func):
                                (batch_x, batch_y) = data

                                batch_x = move_data_to_gpu(batch_x, cuda)

                                model.eval()
                                with torch.no_grad():
                                    y_pred_sigmoid = model(batch_x)
                                    all_batch_y.append(batch_y)
                                    all_pred_y.append(y_pred_sigmoid.data.cpu().numpy())

                            all_pred_y = np.concatenate(all_pred_y)
                            all_batch_y = np.concatenate(all_batch_y)

                            test_acc = accuracy_score(all_batch_y, (all_pred_y > 0.5).astype(float))
                            print('n_test_acc:', test_acc)

                            y_preds_all.append(
                                np.concatenate([1 - all_pred_y, all_pred_y], axis=-1))  # Check ordering of classes (yes/no)

                        y_preds_all = np.array(y_preds_all)
                        print('y_preds_all: ', y_preds_all.shape, all_batch_y.shape)

                        results = {'y_preds_all': y_preds_all, 'all_batch_y': all_batch_y}
                        with open(preditions_file, 'wb') as f:
                            pickle.dump(results, f)
                    else:
                        print('loading: ', preditions_file)
                        with open(preditions_file, 'rb') as input_file:
                            results = pickle.load(input_file)
                            y_preds_all = results['y_preds_all']
                            all_batch_y = results['all_batch_y']

                    plot_dir = os.path.join(workspace, 'plot_' + filename)
                    create_folder(plot_dir)
                    PE, MI, log_prob = get_results(y_preds_all, all_batch_y, plot_dir, filename=filename)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















