import sys, os, argparse, torch
from framework.data_generator import DataGenerator_Yuanbo
from framework.models import *
from framework.processing import training_process
from framework.utilities import create_folder
import framework.config as config

# gpu_id = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(argv):
    model_name = 'MTRCNN'
    # model_name = 'VGGishDropoutFeatB'
    # model_name = 'PANN'
    # model_name = 'MobileNetV2'
    # model_name = 'YAMNet'
    # model_name = 'CNN_Transformer'

    using_model = eval(model_name)

    if model_name == 'VGGishDropoutFeatB':
        model = using_model(preprocess=False, dropout=0.2)
    elif model_name == 'MobileNetV2' or model_name == 'YAMNet' or model_name == 'CNN_Transformer':
        model = using_model(class_num=1, dropout=0.2, MC_dropout=True)
    else:
        model = using_model(class_num=1, dropout=0.2, MC_dropout=True, batchnormal=True)

    params_num = count_parameters(model)
    print('Parameters num: {}'.format(params_num))

    from torchprofile import profile_macs
    input1 = torch.randn(1, 1, 100, config.mel_bins)
    mac = profile_macs(model, input1)
    print('MAC = ' + str(mac / 1000 ** 3) + ' G')

    from thop import profile
    input1 = torch.randn(1, 1, 100, config.mel_bins)
    flops, params = profile(model, inputs=(input1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # GFLOPS （Giga-FLOPS）

    # FLOPs = 0.07601024G
    # Params = 0.239441M


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















