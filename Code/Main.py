import argparse
import numpy as np
import os
from Code.ELDB import ELDB


def get_parser():
    """默认参数设置"""
    ini = np.recfromtxt("../Data/Ini/parameter_ini.txt")
    save_path_parameter_txt = (str(ini[1][-1], encoding="utf-8") + data_name + '_' +
                               mode_action + '_' + str(ini[2][-1], encoding="utf-8") + ".txt")
    if not os.path.exists(save_path_parameter_txt):
        print(save_path_parameter_txt)
        psi = 0.9
        type_b2b = "ave"
        mode_bag_init = 'g'
    else:
        parameters = np.recfromtxt(save_path_parameter_txt)
        psi = float(parameters[0][-1])
        type_b2b = str(parameters[1][-1], encoding="utf-8")
        mode_bag_init = str(parameters[2][-1], encoding="utf-8")

    parser = argparse.ArgumentParser(description="多示例学习ELDB算法的参数设置")
    parser.add_argument("--psi", default=psi, choices=np.arange(0.1, 1.1, 0.1), help="辨别包的选取比例")
    parser.add_argument("-alpha", default=0.75, type=float, help="学习率")
    parser.add_argument("--psi_max", default=100, type=int, help="最大选取包数")
    parser.add_argument("--type_b2b", default=type_b2b, help="距离度量")
    parser.add_argument("--mode_bag_init", default=mode_bag_init, help="初始dBagSet选取模式")
    parser.add_argument("--mode-action", default=mode_action, help="行为模式")
    parser.add_argument("--k", default=int(str(ini[2][-1], encoding="utf-8")))
    parser.add_argument("--type_performance", default=["f1_score"], type=list, help="性能度量指标")
    parser.add_argument("--save_path_classification_result", default=str(ini[0][-1], encoding="utf-8"),
                        help="分类结果保存路径")
    parser.add_argument("--print_loop", action="store_false", default=False, help="是否打印loop变化值")

    return parser.parse_args()


def main():
    """
    测试
    """
    args = get_parser()
    eldb = ELDB(data_path=data_path, psi=args.psi, alpha=args.alpha, psi_max=args.psi_max,
                type_b2b=args.type_b2b, mode_bag_init=args.mode_bag_init,
                mode_action=args.mode_action, k=args.k,
                type_performance=args.type_performance, print_loop=args.print_loop)
    results = {}
    results_save = {}
    classifier_type, metric_type = eldb.get_state()
    # 获取完整的CV实验结果
    for i in range(10):
        result_temp = eldb.get_mapping()
        for classifier in classifier_type:
            for metric in metric_type:
                val_temp = float("{:.4f}".format(result_temp[classifier + ' ' + metric] * 100))
                if i == 0:
                    results[classifier + ' ' + metric] = [val_temp]
                else:
                    results[classifier + ' ' + metric].append(val_temp)
    # 计算平均值以及标准差
    for metric in metric_type:
        best_ave[metric] = 0
        for i, classifier in enumerate(classifier_type):
            key = classifier + ' ' + metric
            ave_temp = float("{:.4f}".format(np.average(results[key])))
            std_temp = float("{:.4f}".format(np.std(results[key], ddof=1)))

            # 记录最佳结果
            if ave_temp > best_ave[metric]:
                best_classifier[metric] = classifier
                best_ave[metric] = ave_temp
                best_std[metric] = std_temp
                results_save[metric] = results[key]

    # 如果是最佳结果则进行保存
    data_save_path = (args.save_path_classification_result + data_name + '_' +
                      args.mode_action + '_' + str(args.k) + ".npz")
    if not os.path.exists(data_save_path):
        np.savez(data_save_path, best_classifier=best_classifier, best_ave=best_ave, best_std=best_std,
                 results_save=results_save)
    best_results_load = np.load(data_save_path, allow_pickle=True)
    # 输出最佳分类结果
    print("最佳分类结果 (行为模式{}, {}折交叉验证)：".format(args.mode_action, args.k))
    for metric in metric_type:
        if best_ave[metric] < eval(str(best_results_load["best_ave"]))[metric]:
            best_classifier[metric] = eval(str(best_results_load["best_classifier"]))[metric]
            best_ave[metric] = eval(str(best_results_load["best_ave"]))[metric]
            best_std[metric] = eval(str(best_results_load["best_std"]))[metric]
            results_save[metric] = eval(str(best_results_load["results_save"]))[metric]
        print("\t{:s}度量{:s}分类器结果：".format(metric, best_classifier[metric]))
        print("\tk次验证结果：", results_save[metric])
        print("\t平均分类精度及标准差：", best_ave[metric], best_std[metric])
    np.savez(data_save_path, best_classifier=best_classifier, best_ave=best_ave, best_std=best_std,
             results_save=results_save)


if __name__ == '__main__':
    """进行实验时需要修改的参数"""
    # 数据集的路径
    data_path = "../Data/Benchmark/musk1+.mat"
    # 行为模式，对应于aELDB和rELDB
    mode_action = 'r'  # or 'r'

    # 用于记录最佳分类器的分类结果
    best_classifier, best_ave, best_std = {}, {}, {}
    # 获取数据集名称
    data_name = data_path.split('/')[-1].split('.')[0]
    print("实验数据集{:s}".format(data_name))
    main()
