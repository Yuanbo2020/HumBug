import sys, os, argparse, torch
from framework.data_generator import DataGenerator_Yuanbo
from framework.models import *
from framework.processing import training_process
from framework.utilities import create_folder
import framework.config as config

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
    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数解析
    # parser.add_argument('-alpha', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('-adamw', action="store_true")
    parser.add_argument('-check_step', type=int, required=True)
    parser.add_argument('-repeat_no', type=int, required=True)
    parser.add_argument('-lr_rate', type=float, required=True)
    parser.add_argument('-batch', type=int, required=True)
    parser.add_argument('-epochs', type=int, required=True)
    parser.add_argument('-model_name', type=str, required=True)

    # parser.add_argument('-monitor', type=str, required=True)
    # parser.add_argument('-early_stopping_patience', type=int, required=True)
    # parser.add_argument('-early_stopping_warmup', type=int, required=True)
    # parser.add_argument('-alpha', nargs='+', help='<Required> Set flag', required=True)
    # 这样的话，当你不输入-lr_decay的时候，默认为False；输入-lr_decay的时候，才会触发True值
    # 开始解析
    args = parser.parse_args()

    repeat_no = args.repeat_no
    lr_init = args.lr_rate  # 1e-3
    batch_size = args.batch
    epochs = args.epochs
    model_name = args.model_name
    adamw = args.adamw
    # check_step = args.check_step

    sys_name = model_name + '_' + str(repeat_no) + '_lr' + str(lr_init).replace('-', '') + '_e' + str(epochs) + '_b' + str(batch_size)
    workspace = os.path.join(os.getcwd(), sys_name)

    repeat = False  # True  # False
    if repeat or not os.path.exists(workspace):
        create_folder(workspace)

        log_path = os.path.join(workspace, 'logs')
        create_folder(log_path)
        filename = os.path.basename(__file__).split('.py')[0]
        print_log_file = os.path.join(log_path, filename + '_print.log')
        sys.stdout = Logger(print_log_file, sys.stdout)
        console_log_file = os.path.join(log_path, filename + '_console.log')
        sys.stderr = Logger(console_log_file, sys.stderr)

        generator = DataGenerator_Yuanbo(batch_size)

        using_model = eval(model_name)

        if model_name == 'VGGishDropoutFeatB':
            model = using_model(preprocess=False, dropout=0.2)
        elif model_name == 'MobileNetV2' or model_name == 'YAMNet' or model_name == 'CNN_Transformer':
            model = using_model(class_num=1, dropout=0.2, MC_dropout=True)
        else:
            model = using_model(class_num=1, dropout=0.2, MC_dropout=True, batchnormal=True)

        if config.cuda and torch.cuda.is_available():
            model.cuda()

        models_dir = os.path.join(workspace, 'model')
        training_process(generator, model, models_dir, epochs, batch_size, lr_init=lr_init, log_path=log_path, adamw=adamw)
        print('Training is done!!!')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















