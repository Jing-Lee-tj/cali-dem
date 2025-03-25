# -*- coding: utf-8 -*-
import os
import shutil
import time
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor

from cali_setting import load_input # 加载校正设置
from cali_infrastructure import (          
    generate_exper_plan,    # 生成实验计划
    split_to_cpu,           # 根据 CPU 设置划分任务
    split_ex_plan,          # 分割实验计划
    start,                  # 执行单次 DEM 模拟
    cost_function,          # 误差函数
    evaluate_agent,         # 代理模型评估函数
    evaluate_dem,           # DEM 模型评估函数
    nonlin_residmin,        # 非线性残差最小化优化函数
    create_agent_models,    # 创建代理模型
    get_optim_folder,       # 获取优化结果存放文件夹的名称
)

def run_model(workdir, models, run_params):
    return start(workdir, models, run_params)

def main():
    """ 
    主函数： 执行实验设计、并行计算、代理模型和优化计算
    """
    start_time = time.time()
    workdir = os.getcwd()

    # --------------------- 初始化及环境设置 ---------------------
    with open("logfile.txt", "w") as log_file:
        log_file.write("Starting the process...\n")

        # 清理并创建必要的文件夹
        for folder in ['optim', 'results', 'AgentFuns']:
            folder_path = os.path.join(workdir, folder)
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path, exist_ok=True)

        # 加载校正设置
        Input, optim_settings, _, _, paramLims = load_input()

        # --------------------- 实验计划 ---------------------
        runs = generate_exper_plan(workdir, Input['numOfSam'], paramLims)
        results_all, NaN_runs, foldername = {}, {}, []

        # --------------------- DEM 模拟 ---------------------
        for k, model in enumerate(Input['model']):
            print(f"Starting DEM-Simulation for {model}")
            log_file.write(f"Starting DEM-Simulation for {model}\n")

            # 为当前模型创建实验目录
            model_folder = os.path.join(workdir, 'optim', model)
            os.makedirs(model_folder, exist_ok=True)

            # CPU 分配
            numProc, runs_p_CPU, remainder = split_to_cpu(Input['maxCPU'], len(runs), Input['cpu'][k])
            runs_all_list = split_ex_plan(runs, numProc, runs_p_CPU, remainder)

            # 附加模型索引列
            runs_all_list = [np.hstack((arr, np.full((arr.shape[0], 1), k))) for arr in runs_all_list]

            # 运行 DEM 模型（并行计算）
            workers = int(Input['maxCPU'] / Input['cpu'][k])
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(run_model, itertools.repeat(workdir), itertools.repeat(Input['model']), runs_all_list))
            
            # 运行 DEM 模型（顺序计算，用于调试）
            # results = []
            # for item in runs_all_list:
            #     resutls = results.append([run_model(workdir, Input['model'], item)])

            results_all[model] = np.vstack(results)

            # 记录包含 NaN 结果的运行索引
            NaN_runs[model] = [idx for idx, result in enumerate(results) if np.isnan(result).any()]
            for idx in NaN_runs[model]:
                print(f"Warning: NaN run at index {idx}")
                log_file.write(f"Warning: NaN run at index {idx}\n")

        # --------------------- 处理 NaN 运行 & 保存结果 ---------------------
        all_nan_indices = sorted(set(idx for indices in NaN_runs.values() for idx in indices))
        runs_DoE = np.delete(runs, all_nan_indices, axis=0)
        np.savetxt(os.path.join(workdir, "ExperimentalPlan_DoE.csv"), runs_DoE, delimiter=",")

        # 逐个保存去除 NaN 后的实验结果
        for model in Input['model']:
            cleaned_results = np.delete(results_all[model], all_nan_indices, axis=0).T
            for i, result in enumerate(cleaned_results):
                folder_name = f"DoE_{model}_{i}"
                folder_path = os.path.join(workdir, "results", folder_name)
                os.makedirs(folder_path, exist_ok=True)
                np.savetxt(os.path.join(folder_path, f"{folder_name}.csv"), result.T, delimiter=",")
                foldername.append(folder_name)

        # --------------------- 代理建模 ---------------------
        ExPath = [os.path.join(workdir, "ExperimentalPlan_DoE.csv")] * len(foldername)
        RePath = [os.path.join(workdir, "results", name, f"{name}.csv") for name in foldername]

        # 目标值检查
        for i, repath in enumerate(RePath):
            try:
                tmp = np.loadtxt(repath, delimiter=",")
                if not (tmp.min() <= optim_settings['targetVal'][i] <= tmp.max()):
                    print(f"WARNING: Target value {optim_settings['targetVal'][i]} out of range in {repath}.")
            except Exception as e:
                print(f"Error reading {repath}: {e}")
                log_file.write(f"Error reading {repath}: {e}\n")

        # 创建 代理 模型
        print("Creating Agent models...")
        AgentFunNames = create_agent_models(workdir, ExPath, RePath, paramLims)

        # --------------------- 代理优化 ---------------------
        costFunc = lambda params: cost_function(params, evaluate_agent, workdir, AgentFunNames)
        settings = {
            'lbound': paramLims['min'], 'ubound': paramLims['max'],
            'MaxFunEvals': 3000, 'TolFun': optim_settings["tolfun"],
            'tolRes': optim_settings["tolRes"],
        }
        p, _, _, _ = nonlin_residmin(costFunc, (paramLims['min'] + paramLims['max']) / 2, settings)

        # --------------------- DEM 模型优化 ---------------------
        costFunc = lambda params: cost_function(params, evaluate_dem, workdir)
        settings['MaxFunEvals'] = optim_settings["maxFunEvals"]
        p_DEM, _, _, _ = nonlin_residmin(costFunc, p, settings)

        # --------------------- 输出优化结果 ---------------------
        elapsed_time = time.time() - start_time
        optimFolderName = get_optim_folder(p_DEM, workdir)

        print("Results are stored in:")
        log_file.write("Results are stored in:\n")
        for model, folder in zip(Input['model'], optimFolderName):
            if optimFolderName:
                result_path = os.path.join(workdir, 'optim', model, folder)
                print(result_path)
                log_file.write(result_path + "\n")

        print(f"Finished in {elapsed_time / 60:.2f} minutes")
        log_file.write(f"Finished in {elapsed_time / 60:.2f} minutes\n")

if __name__ == '__main__':
    main()

