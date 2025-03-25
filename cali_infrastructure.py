# -*- coding: utf-8 -*-
# --------------------------
# 标准库导入
# --------------------------
import csv
import itertools
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
import importlib
from concurrent.futures import ProcessPoolExecutor

# --------------------------
# 第三方库导入
# --------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyDOE2 import lhs
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
import joblib

# --------------------------
# 本地模块导入
# --------------------------
from cali_setting import load_input  # 加载校正设置

def generate_exper_plan(workdir, N, paramLims, num_iter=100):
    lower = paramLims['min']
    upper = paramLims['max']
    
    DIM = len(lower)
    NITER = num_iter * DIM
    
    lhs_design = lhs(DIM, samples=N, criterion='maximin', iterations=NITER)
    
    ExPlan = lower + (upper - lower) * lhs_design
    
    csv_file = os.path.join(workdir, "ExperimentalPlan.txt")
    np.savetxt(csv_file, ExPlan, delimiter=",")
    
    return ExPlan

def split_to_cpu(maxCPU, nRuns, nCPU_run):
    # 计算可并行运行的 DEM 模型进程数（向下取整）
    numProc = maxCPU // nCPU_run

    # 如果计算得到的进程数大于总运行次数，则限制为 nRuns
    if numProc > nRuns:
        numProc = nRuns

    # 计算每个进程处理的运行次数（整除）
    runs_p_CPU = nRuns // numProc

    # 计算剩余的运行次数（模运算）
    remainder = nRuns % numProc

    # Sanity check：确认分配的运行次数之和与总运行次数相符
    if numProc * runs_p_CPU + remainder != nRuns:
        print("WARNING: Some runs of the experimental plan might not be computed with the DEM model.")
        print("Check the configuration of available CPUs and the CPUs required per DEM model.")
        print("This warning was generated from the function split_to_cpu.")

    return numProc, runs_p_CPU, remainder

def split_ex_plan(ExPlan, nProc, runs_p_CPU, remainder):
    total_rows = ExPlan.shape[0]
    
    # 生成所有行号（1-indexed）
    indices = np.arange(1, total_rows + 1).reshape(-1, 1)
    
    # 将行号追加到数据的最后一列
    ExPlan_with_index = np.hstack((ExPlan, indices))

    # 主部分划分
    runs_main = [ExPlan_with_index[i * runs_p_CPU: (i + 1) * runs_p_CPU] for i in range(nProc)]
    
    # 剩余部分
    runs_rest = [ExPlan_with_index[k:k+1] for k in range(nProc * runs_p_CPU, total_rows)]
    
    # 合并结果（仅在有剩余任务时）
    return runs_main + runs_rest if remainder != 0 else runs_main

def start(workdir, models, run_params):
    
    model_indices = run_params[:, -1].astype(int)  # 确保索引为整数
    
    # 遍历每一组参数，调用 run_script
    results = [run_script(workdir, models[model_idx], params) 
                   for params, model_idx in zip(run_params, model_indices)]

    return np.array(results)

def run_script(workdir, model, params):
    # --------------------- 加载模型变量 ---------------------
    _, _, modelVars, assign, _ = load_input()
    modelVars = assign_params(modelVars, params[:-2], assign)  # 最后一行是实验计划编号

    # --------------------- 创建模拟文件夹 ---------------------
    folder_number = get_last_folders_number(workdir, model)
    new_folder_name = get_new_folder_name(folder_number + 1)

    if create_new_folder(workdir, model, new_folder_name) == 1:
        print(f"{time.ctime()} New folder {new_folder_name} created!")
    else:
        print(f"ERROR in run_script(): Could not create the folder {new_folder_name}!")
        return None  # 直接返回，避免继续执行

    # --------------------- 写入模型配置 ---------------------
    # 写入参数到 CSV 文件
    write_params(workdir, model, new_folder_name, params)

    model_path = os.path.join(workdir, "DEMmodels", model)
    sys.path.append(model_path)

    # --------------------- 启动模拟 ---------------------
    simulation_path = os.path.join(workdir, "optim", model, new_folder_name)
    
    try:
        from shear import run_simulation, get_results
        run_simulation(simulation_path, modelVars)
        print(f"Simulation completed for {new_folder_name}.")
    except Exception as e:
        print(f"ERROR: Unable to run simulation in folder '{new_folder_name}' for model '{model}'.")
        print(f"Reason: {e}")
        sys.path.remove(model_path)
        return None  # 运行失败直接返回，不尝试获取结果

    # --------------------- 处理模拟结果 ---------------------
    try:
        results = get_results(simulation_path)
        print(f"Results successfully retrieved for {new_folder_name}.")
    except Exception as e:
        print(f"ERROR: Unable to retrieve results for folder '{new_folder_name}' in model '{model}'.")
        print(f"Reason: {e}")
        results = None  # 确保返回值存在，即使获取失败

    # 退出时移除路径
    sys.path.remove(model_path)
    
    return results

def assign_params(modelVars, paramVars, assign):
    # 检查 assign 和 paramVars 长度是否匹配
    if len(assign) != len(paramVars):
        raise ValueError("The length of 'assign' must match the length of 'paramVars'.")

    # 更新 modelVars 字典中的值
    for var_name, value in zip(assign, paramVars):
        modelVars[var_name] = value

    # 更新 Rayleigh 时间步和模拟时间步
    modelVars["RLTS"] = get_rayleigh(
        modelVars["radiusP"],
        modelVars["densityP"],
        modelVars["youngsModulusP"],
        modelVars["poissonsRatioP"]
    )
    modelVars["SIMTS"] = modelVars["RLTS"] * modelVars["percentRayleigh"]

    return modelVars

def get_rayleigh(r, rho, YM, nu):
    try:
        # 计算剪切模量
        SM = YM / (2 * (1 + nu))
        
        # 计算 Rayleigh 时间
        dtr = np.pi * r * np.sqrt(rho / SM) / (0.1631 * nu + 0.8766)
        
        # 使用 np.nan_to_num 处理 NaN 和 Inf，默认将它们替换为 0
        return np.nan_to_num(dtr, nan=0, posinf=0, neginf=0)
    
    except Exception as e:
        print(f"Error in get_rayleigh: {e}")
        return None

def get_last_folders_number(workdir, model):
    """
    获取模型文件夹中最新的编号（文件夹名前 4 位数字）。

    参数:
        workdir (str): 工作目录
        model   (str): 模型名称

    返回:
        int: 最新编号，若无有效编号或超过 9000，则返回 0
    """
    last_folder_name = get_last_folders_name(workdir, model)
    if not last_folder_name:
        return 0

    match = re.match(r"(\d{4})", last_folder_name)
    if match:
        number = int(match.group(1))
        if number > 9000:
            print("WARNING: foldersNumber over 9000!")
            return 0
        return number
    return 0
    
def get_last_folders_name(workdir, model):
    """
    获取模型文件夹中最新的文件夹名称。

    参数:
        workdir (str): 工作目录
        model   (str): 模型名称

    返回:
        Optional[str]: 最新文件夹名称，若不存在则返回 None
    """
    model_path = os.path.join(workdir, "optim", model)

    if not os.path.isdir(model_path):
        return None

    try:
        # 获取所有子文件夹
        folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        if not folders:
            return None

        # 按自然排序（数字优先）
        folders.sort(key=lambda x: (x.isdigit(), x))
        return folders[-1]
    except Exception as e:
        print(f"Error in get_last_folders_name: {e}")
        return None

def get_new_folder_name(num):
    """
    生成 `XXXXyyy_XXXXX` 格式的文件夹名称：
      - `XXXX`：4 位数字（补零）。
      - `yyy`：3 个随机小写字母。
      - `XXXXX`：进程 ID。

    参数:
        num (int): 文件夹编号

    返回:
        str: 生成的文件夹名称，例如 '0001abc_12345'
    """
    number = f"{num:04d}"  # 补零格式化
    letters = ''.join(random.choices(string.ascii_lowercase, k=3))
    process_id = os.getpid()

    return f"{number}{letters}_{process_id}"

def create_new_folder(workdir, model, folder_name):
    """
    创建模拟文件夹，并复制 `DEMmodels/model` 到 `optim/model`。

    参数:
        model (str): 模型名称
        folder_name (str): 新文件夹名称
        workdir (str): 工作目录

    返回:
        int: 1 表示成功，0 表示失败
    """
    src_folder = os.path.join(workdir, "DEMmodels", model)
    dest_folder = os.path.join(workdir, "optim", model, folder_name)

    if not os.path.isdir(src_folder):
        print(f"Error: Source folder '{src_folder}' not found.")
        return 0

    os.makedirs(os.path.dirname(dest_folder), exist_ok=True)

    try:
        shutil.copytree(src_folder, dest_folder)
        print(f"Folder created: {dest_folder}")
        return 1
    except FileExistsError:
        print(f"Warning: '{dest_folder}' already exists.")
    except PermissionError:
        print(f"Error: Permission denied for '{dest_folder}'.")
    except Exception as e:
        print(f"Error: {e}")

    return 0

def write_params(workdir, model, folder_name, params):
    # 构建存储路径
    folder_path = os.path.join(workdir, 'optim', model, folder_name)
    params_file_path = os.path.join(folder_path, 'params.txt')
    explan_file_path = os.path.join(folder_path, 'ExPlan.txt')

    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)

    try:
        # 保存参数到 CSV 文件
        np.savetxt(params_file_path, params[:-2], delimiter=',', fmt='%s')
        print(f"Successfully saved parameters to {params_file_path}")

        # 如果 params[-2] 非零，更新 ExPlan 文件
        if params[-2] != 0:
            with open(explan_file_path, 'a', encoding='utf-8') as fd:
                fd.write(f"{int(params[-2])}\n")  # 写入 ExPlan 行号
            print(f"Successfully updated ExPlan file: {explan_file_path}")

    except FileNotFoundError:
        print(f"Error: The directory or file '{params_file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied when writing to '{params_file_path}'.")
    except Exception as e:
        print(f"Unexpected error while writing to files in {folder_path}: {e}")

def create_agent_models(workdir, ex_plan_files, resp_files, paramLims, cov_model='matern'):
    agent_fun_names = []

    # 构造核函数
    if cov_model == 'matern':
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct()) + \
                 (Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2))) + \
                 (WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))
    else:
        raise ValueError(f"Unsupported covariance model: {cov_model}")

    # 校验数据一致性
    for exp_file, resp_file in zip(ex_plan_files, resp_files):
        try:
            ex_plan = np.genfromtxt(exp_file, delimiter=',')
            resp = np.genfromtxt(resp_file, delimiter=',')

            if ex_plan.shape[1] != len(paramLims['min']):
                print(f"WARNING: Parameter count mismatch ({paramLims.shape[1]} vs {ex_plan.shape[1]}).")

            if len(resp) != ex_plan.shape[0]:
                print(f"WARNING: Response length mismatch for {resp_file}.")

        except Exception as e:
            print(f"Error reading files: {exp_file} or {resp_file}. Error: {e}")
            return None

    # 代理模型存储路径
    agent_folder_path = os.path.join(workdir, "AgentFuns")
    if os.path.exists(agent_folder_path):
        shutil.rmtree(agent_folder_path)
    os.makedirs(agent_folder_path, exist_ok=True)

    # 创建代理模型
    for i, (exp_file, resp_file) in enumerate(zip(ex_plan_files, resp_files)):
        try:
            ex_plan = np.genfromtxt(exp_file, delimiter=',')
            resp = np.genfromtxt(resp_file, delimiter=',')

            # 归一化参数
            lower = paramLims['min']
            upper = paramLims['max']
            ex_plan_norm = (ex_plan - lower) / (upper - lower)

            # 训练代理模型
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
            gpr.fit(ex_plan_norm, resp)

            model_name = f"Agent_fun_{i+1}"
            agent_fun_names.append((model_name, resp_file))

            # 生成代理模型函数文件
            create_agent_fun(workdir, model_name, gpr, ex_plan, resp, paramLims)

        except Exception as e:
            print(f"Error creating agent model for {resp_file}: {e}")

    return agent_fun_names

def create_agent_fun(workdir, fun_name, model, ex_plan, resp, paramLims):
    agent_folder = os.path.join(workdir, 'AgentFuns')
    os.makedirs(agent_folder, exist_ok=True)

    # 保存模型数据文件
    model_filename = f"{fun_name}.model"
    model_data_path = os.path.join(agent_folder, model_filename)

    try:
        with open(model_data_path, 'wb') as f:
            joblib.dump({'model': model, 'ExPlan': ex_plan, 'Response': resp, 'paramLims': paramLims}, f)
        print(f"Saved agent model: {model_data_path}")
    except Exception as e:
        print(f"Error saving agent model '{fun_name}': {e}")
        return 0

    # 生成代理模型函数文件
    func_filename = f"{fun_name}.py"
    func_filepath = os.path.join(agent_folder, func_filename)

    try:
        with open(func_filepath, 'w', encoding="utf-8") as f:
            f.write(f"""
import numpy as np
import joblib
import os

def {fun_name}(x) -> np.ndarray:
    \"\"\" 
    载入代理模型并进行预测。 
    
    参数:
        x (array-like): 输入参数数组，形状为 (num_params,)。
    
    返回:
        np.ndarray: 预测结果（均值）。
    \"\"\"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_data_path = os.path.join(script_dir, '{model_filename}')
    
    try:
        model_data = joblib.load(model_data_path)
        model = model_data['model']
        paramLims = model_data['paramLims']
    except Exception as e:
        raise RuntimeError(f"Error loading agent model: {{e}}")
    
    lower = paramLims['min']
    upper = paramLims['max']
    x_norm = (np.array(x) - lower) / (upper - lower)

    mean = model.predict(np.array([x_norm]))[0]
    return mean
""")
        print(f"Saved agent function: {func_filepath}")
    except Exception as e:
        print(f"Error writing agent function '{fun_name}': {e}")

def evaluate_agent(params, workdir, agent_fun_names):
    results = []
    agent_path = os.path.join(workdir, 'AgentFuns')

    if not os.path.isdir(agent_path):
        print(f"Error: Agent functions directory '{agent_path}' does not exist.")
        return np.full(len(agent_fun_names), np.nan)

    original_sys_path = list(os.sys.path)
    if agent_path not in os.sys.path:
        os.sys.path.append(agent_path)

    try:
        for fun_name, _ in agent_fun_names:
            try:
                agent_module = importlib.import_module(fun_name)
                agent_predict_func = getattr(agent_module, fun_name)

                mean = agent_predict_func(params)
                results.append(mean)
            except ModuleNotFoundError:
                print(f"Error: Agent function module '{fun_name}' not found.")
                results.append(np.nan)
            except AttributeError:
                print(f"Error: Agent function '{fun_name}' not callable.")
                results.append(np.nan)
            except Exception as e:
                print(f"Unexpected error evaluating agent function '{fun_name}': {e}")
                results.append(np.nan)
    finally:
        os.sys.path = original_sys_path

    return np.array(results)

def evaluate_dem(params, workdir):
    from calibration import run_model

    try:
        # 加载输入，假设 load_input() 返回包含模型信息的对象或字典
        Input, _, _, _, _ = load_input()

        # 生成 evalRuns，每个元素为 params 的副本，追加 0 和模型编号（1-indexed）
        evalRuns = [
            np.hstack((np.array(params).reshape(1, -1), [[0, i],]))
            for i in range(len(Input['model']))
        ]

        # 判断可用 CPU
        available = Input['maxCPU'] - sum(Input['cpu'])
        if available > 0:
            workers = Input['maxCPU'] // np.max(Input['cpu'])

            # 并行执行
            model_list = itertools.repeat(Input['model'], len(evalRuns))
            workdir_list = itertools.repeat(workdir, len(evalRuns))

            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(run_model, workdir_list, model_list, evalRuns))
        else:
            # 顺序执行
            results = [run_model(run, Input['model'], workdir) for run in evalRuns]

        return np.array(results).reshape(-1, 1)

    except Exception as e:
        print(f"Error in evaluate_dem: {e}")
        return 0

def get_min_max_rayleigh(modelVars: dict) -> dict:
    """
    计算给定参数空间的 Rayleigh 时间的最小/最大值。

    参数:
        modelVars (dict): 包含当前模型变量的字典。

    返回:
        dict: RLTS，包含 Rayleigh 时间的最小值和最大值，格式 {'min': value, 'max': value}
    """
    try:
        # 加载模型输入，假设返回值结构与原代码相同
        _, _, modelVars, assign, paramLims = load_input()

        # 简化获取参数最小/最大值的过程
        def get_param_minmax(param_name, modelVars: dict) -> np.ndarray:
            """ 获取参数的最小/最大值，如果未指定，则使用默认值 """
            if param_name in assign:
                index = assign.index(param_name)
                return paramLims[:, index]
            return np.array([modelVars[param_name], modelVars[param_name]])

        # 获取参数的最小/最大值
        radiusP_minmax = get_param_minmax('radiusP', modelVars)
        rhoP_minmax = get_param_minmax('densityP', modelVars)
        nuP_minmax = get_param_minmax('poissonsRatioP', modelVars)

        # 获取杨氏模量的最小/最大值，处理缺失的情况
        if 'youngsModulusP' in assign:
            YM_minmax = np.flip(paramLims[:, assign.index('youngsModulusP')])
        else:
            YM_minmax = np.array([modelVars['youngsModulusP'], modelVars['youngsModulusP']])

        # 计算 Rayleigh 时间
        rayleigh_times = get_rayleigh(radiusP_minmax, rhoP_minmax, YM_minmax, nuP_minmax)

        return {'min': rayleigh_times[0], 'max': rayleigh_times[1]}

    except Exception as e:
        print(f"Error in get_min_max_rayleigh: {e}")
        return {'min': np.nan, 'max': np.nan}
    
def get_optim_folder(workdir, optim_params):
    try:
        Input, _, _, _, _ = load_input()  # 加载输入
        folder_names = []

        # 遍历每个模型
        for model in Input['model']:
            model_folder = os.path.join(workdir, 'optim', model)
            
            # 如果模型文件夹不存在，跳过
            if not os.path.exists(model_folder):
                folder_names.append(np.nan)
                continue
            
            # 查找符合条件的文件夹
            found_folder = find_matching_folder(model_folder, optim_params)
            folder_names.append(found_folder)

        return folder_names

    except Exception as e:
        print(f"Error in get_optim_folder: {e}")
        return None

def find_matching_folder(workdir, params):
    # 获取模型文件夹内的所有内容
    folder_content = os.listdir(workdir)

    # 遍历每个子文件夹查找符合条件的文件夹
    for folder in folder_content:
        folder_path = os.path.join(workdir, folder)
        params_file = os.path.join(folder_path, 'params.txt')

        # 如果 params.txt 存在，检查其内容是否匹配
        if os.path.exists(params_file):
            try:
                params = np.genfromtxt(params_file, delimiter=',')
                if np.allclose(params, params):
                    return folder
            except Exception as e:
                print(f"Error reading {params_file}: {e}")
    return None

def nonlin_residmin(costFunc, initParam, settings):
    try:
        result = least_squares(
            costFunc,
            initParam,
            bounds=(settings['lbound'], settings['ubound']),
            ftol=settings['TolFun'],  # 误差容忍度
            max_nfev=settings['MaxFunEvals'],  # 最大函数评估次数
        )

        # 提取优化结果
        p_DEM = result.x  # 最优参数
        resid_DEM = result.fun  # 残差（误差）
        cvg_DEM = result.success  # 是否收敛
        outp_DEM = result  # 完整优化结果

        # 检查误差的绝对值是否小于设定的阈值
        if np.all(np.abs(resid_DEM[:-1]) < settings['tolRes']): #最后一个残差为加权瑞丽时间
            print("Error is below the threshold, stopping optimization.")
            return p_DEM, resid_DEM, cvg_DEM, outp_DEM  # 返回优化结果

        return p_DEM, resid_DEM, cvg_DEM, outp_DEM

    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None, False, None

def cost_function(params, evaluate_func, *args):
    
    result_mean = evaluate_func(params, *args)

    # 加载模型变量
    _, optim, model_vars, assign, paramLims = load_input()
    model_vars = assign_params(model_vars, params, assign)

    # 计算 Rayleigh 时间步
    RLTS = get_min_max_rayleigh(model_vars)

    # 计算残差
    residual = (result_mean - optim['targetVal']) / optim['targetVal']  # 目标值误差

    # 加权 Rayleigh 时间步差异
    if not np.isclose(RLTS['max'], RLTS['min']):
        residual = np.append(residual, optim['WRL'] * ((RLTS['max'] - model_vars['RLTS']) / (RLTS['max'] - RLTS['min'])))
    else:
        residual = np.append(residual, 0)

    return residual





