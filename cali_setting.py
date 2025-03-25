import numpy as np

def load_input():
    # --------------------- 实验计划 ---------------------
    Input = {
        "model": ["shear10", "shear50", "shear250"],
        "cpu": [1,1,1],   # 每个模型运行时所使用的CPU 数量
        "maxCPU": 24,  # 最大可用 CPU 资源
        "numOfSam": 200,         # 拉丁超立方采样的样本数
        } 
    
    # --------------------- 优化相关参数 ---------------------
    optim = {
        # 目标值，对应get_results的返回变量
        "targetVal": np.array([  
            [444770, 5, 0.382],
            [2447768, 21, 0.309],
            [10711803, 77, 0.216],
        ]),
        # 误差
        "tolRes": np.array([
            [0.01, 0.01, 0.01],
        ]),
        "tolfun": 0.001,      # 函数容忍度
        "maxFunEvals": 50,    # 最大函数评估次数
        "WRL": 0.5,           # Rayleigh 时间步的权重因子
    }

    # --------------------- 固定模型变量 ---------------------
    modelVars = {
        "poissonsRatioP": 0.30,
        "radiusP": 0.003,
        "youngsModulusP": 50e6,
        "percentRayleigh": 0.35,
        "densityP": 2500,
    }

    # --------------------- 校准变量 & 参数边界 ---------------------
    assign = ["kn", "gamman", "xmu"]  

    paramLims = {
        "min": np.array([20e6, 500, 0]),
        "max": np.array([56e6, 1500, 1]),
        }
    
    optim["targetVal"] = np.array(optim["targetVal"]).flatten()
    optim["tolRes"] = np.array(optim["tolRes"]).flatten()

    return Input, optim, modelVars, assign, paramLims



