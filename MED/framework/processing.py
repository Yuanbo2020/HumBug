import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models import move_data_to_gpu
import framework.config as config
from sklearn import metrics
from sklearn.metrics import accuracy_score


def forward(model, generate_func, cuda, return_names = False):
    output = []
    label = []

    audio_names = []
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        (batch_x,  batch_y) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            output_sigmoid = model(batch_x)

            output.append(output_sigmoid.data.cpu().numpy())
            # ------------------------- labels -------------------------------------------------------------------------
            label.append(batch_y)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)

    dict['prediction'] = np.concatenate(output, axis=0)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label'] = np.concatenate(label, axis=0)
    return dict


def evaluate(model, generate_func, cuda):
    # Forward
    dict = forward(model=model, generate_func=generate_func, cuda=cuda)
    # print(dict['label'].shape, dict['prediction'].shape)  # (1700, 1) (1700, 1)
    val_acc = accuracy_score(dict['label'], (dict['prediction'] > 0.5).astype(float))
    return val_acc


def training_process(generator, model, models_dir, epochs, batch_size, lr_init=1e-3,
                                               log_path=None, adamw=False, cuda=1):
    create_folder(models_dir)

    # Optimizer
    if adamw:
        optimizer = optim.AdamW(model.parameters(), lr=lr_init)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_init)

    BCE_loss = F.binary_cross_entropy

    # max_testing_acc = 0.000001
    # max_testing_acc_itera = 0
    # save_testing_best = 0
    # list_test_acc = []
    list_testing_acc_file = os.path.join(log_path, 'testing_acc.txt')

    max_validation_acc = 0.000001
    max_validation_acc_itera = 0
    save_validation_best = 0
    list_val_acc = []
    list_val_acc_file = os.path.join(log_path, 'val_acc.txt')

    # ------------------------------------------------------------------------------------------------------------------

    sample_num = len(generator.y_train)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = one_epoch
    print('validating every: ', check_iter, ' iteration')

    training_start_time = time.time()
    overrun_counter = 0
    break_flag = False

    for iteration, all_data in enumerate(generator.generate_training()):
        (batch_x, batch_y) = all_data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y_cpu = batch_y
        batch_y = move_data_to_gpu(batch_y, cuda)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        batch_pre_sigmoid = model(batch_x)
        # print(batch_pre_sigmoid.size(), batch_y.size())  # torch.Size([32, 1]) torch.Size([32, 1])
        # print(batch_pre_sigmoid.dtype, batch_y.dtype)  # torch.float32 torch.float32
        loss = BCE_loss(batch_pre_sigmoid, batch_y)

        loss.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        batch_pre_sigmoid = batch_pre_sigmoid.data.cpu().numpy()
        train_acc = accuracy_score(batch_y_cpu, (batch_pre_sigmoid > 0.5).astype(float))
        # all_train_acc.append(train_acc)

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.6f' % float(loss), 'Train_acc: %.6f' % float(train_acc))

        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function

            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)
            list_val_acc.append(val_acc)

            validation_time = time.time() - train_fin_time
            if val_acc > max_validation_acc:
                max_validation_acc = val_acc
                save_validation_best = 1
                max_validation_acc_itera = Epoch
                overrun_counter = -1

            print('E: ', '%.3f' % (Epoch), 'val_acc: %.3f' % float(val_acc))

            print('E: {}, T_validation: {:.3f} s, max_validation_acc: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch), validation_time, max_validation_acc, max_validation_acc_itera))

            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')

            if save_validation_best:
                save_validation_best = 0
                save_out_dict = model.state_dict()
                save_out_path = os.path.join(models_dir, 'best_val' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best validation model saved to {}'.format(save_out_path))

            # 从没有max_validation_acc的轮数开始，max_overrun=10，但这样的自信，来自于验证集和测试集相似
            overrun_counter += 1
            print(
                'Epoch: %d, Train Acc: %.8f, Val Acc: %.8f, overrun_counter %i' % (
                    Epoch, train_acc, val_acc, overrun_counter))

            train_time = train_fin_time - train_bgn_time
            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            # ------------------------ validation done ----------------------------------------------------------------

            # # -------------------------each epoch testing--------------------------------------------------------------
            # print('----------------------evaluating--------------------------------')
            # generate_func = generator.generate_testing(data_type='testing')
            # test_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)
            # list_test_acc.append(test_acc)
            #
            # testing_time = time.time() - train_fin_time
            # if test_acc > max_testing_acc:
            #     max_testing_acc = test_acc
            #     save_testing_best = 1
            #     max_testing_acc_itera = Epoch
            #     overrun_counter = -1
            #
            # print('E: ', '%.3f' % (Epoch), 'test_acc: %.3f' % float(test_acc))
            #
            # print('E: {}, T_testing: {:.3f} s, max_testing_acc: {:.3f} , itera: {} '
            #       .format('%.4f' % (Epoch), testing_time, max_testing_acc, max_testing_acc_itera))
            #
            # np.savetxt(list_testing_acc_file_total, list_test_acc, fmt='%.5f')
            # # np.savetxt(list_val_acc_file_total, list_val_acc, fmt='%.5f')
            #
            # if save_testing_best:
            #     save_testing_best = 0
            #     save_out_dict = model.state_dict()
            #     save_out_path = os.path.join(models_dir, 'best_test' + config.endswith)
            #     torch.save(save_out_dict, save_out_path)
            #     print('Best scene model saved to {}'.format(save_out_path))

            # 从没有max_testing_acc的轮数开始，max_overrun=10，但这样的自信，来自于验证集和测试集相似
            # overrun_counter += 1
            # print(
            #     'Epoch: %d, Train Acc: %.8f, Val Acc: %.8f, overrun_counter %i' % (
            #     Epoch, train_acc, val_acc, overrun_counter))

        if overrun_counter > config.max_overrun:
            break_flag = True

        if iteration > (epochs * one_epoch):
            break_flag = True

        if break_flag:
            finish_time = time.time() - training_start_time
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print("All epochs are done.")

            # correct
            save_out_dict = model.state_dict()
            save_out_path = os.path.join(models_dir, 'final_model' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))
            print('Model training finish time: {:.3f} s,'.format(finish_time))

            print('Training is done!!!')

            print('E: {}, T_validation: {:.3f} s, max_validation_acc: {:.3f} , itera: {} '
                  .format('%.4f' % (Epoch), validation_time, max_validation_acc, max_validation_acc_itera))

            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')

            print('Training is done!!!')

            print('----------------------evaluating--------------------------------')
            # 最终保存的模型

            generate_func = generator.generate_validate(data_type='validate')
            val_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)
            list_val_acc.append(val_acc)

            test_acc_list = []
            test_type = 'test_A'
            generate_func = generator.generate_testing(data_type='test', test_type=test_type)
            test_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)
            test_acc_list.append(test_acc)

            test_type = 'test_B'
            generate_func = generator.generate_testing(data_type='test', test_type=test_type)
            test_acc = evaluate(model=model, generate_func=generate_func, cuda=cuda)
            test_acc_list.append(test_acc)

            np.savetxt(list_testing_acc_file, test_acc_list, fmt='%.5f')
            np.savetxt(list_val_acc_file, list_val_acc, fmt='%.5f')
            break





