# # # -*- coding: utf-8 -*-
# # """
# # Created on Wed Mar 12 14:45:01 2025

# # @author: jingli
# # 2025/3/13 优化了类的封装
# # """
# 标准库
import os
import shutil
import subprocess

import math
import copy
import itertools

import concurrent.futures
from concurrent.futures import as_completed
import threading

# 第三方库
import numpy as np
from scipy.stats import norm

import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 必须在 pyplot 之前
import matplotlib.pyplot as plt


lock = threading.Lock()
max_jobs = 12 #gpu最大线程数为32个
executor1 = concurrent.futures.ThreadPoolExecutor()
executor2 = concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) #实际计算任务提交器

def lmp_run_command(filepath):
    """
    根据输入文件提交lammps运算
    """
    if not os.path.exists(filepath):
        print(f"file {filepath} do not exist")
        return

    workdir = os.path.dirname(filepath)
    
    with lock:
        original_dir = os.getcwd()
        os.chdir(workdir)
        print(f"submit file {filepath}")
        try:
            command = f'lmp -in {filepath}'
            # command = f'lmp -in {filepath} -sf gpu -pk gpu 1'
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            os.chdir(original_dir)
            
    stdout, stderr = process.communicate()

    return filepath, process.returncode, stdout, stderr

# # 基础类设计
# # 文件名规定
# # 加载输入设置
# # 后处理函数等
class BaseSimulation:
    gen_filename = r"generate_packing.bishear"
    con_filename = r"confine_granular_layer.bishear"
    slide_filename = r"steady_slide.bishear"
    vs_filename = r"velocity_step.bishear"
    nss_filename = r"stress_step.bishear"
    nso_filename = r"stress_oscillation.bishear"
    load_con_filename = "load_condition.txt"
    
    #默认参数设置
    params = {
        'seed': 14499,
        'lenx': 0.06,
        'leny': 0.06,
        'kn': 37e6,
        'kt': 45e6,
        'gamman': 1000,
        'gammat': 500,
        'xmu': 0.5,
        'kroll': 9e9,
        'gammaroll': 5000,
        'muroll': 0.2,
        'load_conditions': {
            "icon": [0.25,],
            "ivel": [0.1,],
            "fcon": [0.2,],
        },
        "k_s": 1e9,
    }
    
    
    def __init__(self, workdir, params = dict(), datadir = "restart_data", norun=False):
        self.assign(params)
        self.workdir = os.path.join(workdir,"dem_run",f"sample_{self.params['seed']}")
        os.makedirs(self.workdir, exist_ok=True)
        self.datadir = os.path.join(workdir, datadir)
        os.makedirs(self.datadir, exist_ok=True)
        self.norun=norun
        
        self.set_loading(self.params['load_conditions'])
        
        df = pd.json_normalize(self.params, sep='_')
        df.to_csv(os.path.join(self.workdir,"params.txt"),index=False,header=True)
    
    def assign(self,params):
        self.params.update(copy.deepcopy(params))
        
    def set_loading(self,load_conditions):
    # 设置模型加载工况
        self.con_list = [{"icon": icon} for icon in load_conditions["icon"]]
        self.slide_list = [{"icon": icon, "ivel": ivel} for icon, ivel in itertools.product(load_conditions["icon"], load_conditions["ivel"])]
        
        required_keys = ["icon", "ivel", "fcon"]
        if all(key in load_conditions for key in required_keys):
            self.nss_list = [{"icon": icon, "ivel": ivel, "fcon": fcon} for icon, ivel, fcon in itertools.product(load_conditions["icon"], load_conditions["ivel"], load_conditions["fcon"])]
            
# #     @staticmethod
    def particle_grade_normal(self,Dmean,Dstd,Dmin,Dmax,num_points):
    # 生成正态分布的粒子直径，用于lammps的fix pour命令
        from decimal import Decimal

        num = Decimal(str(Dmean))  
        decimal_places = abs(num.as_tuple().exponent) + 1
        
        dia = np.linspace(Dmin, Dmax, num_points)
        inter = dia[1] - dia[0]
        upboun = dia + inter/2
        downboun = dia - inter/2
        p = norm.cdf(upboun, loc=Dmean, scale=Dstd) - norm.cdf(downboun, loc=Dmean, scale=Dstd)
        p = np.round(p, 4)
        diff = 1 - np.sum(p)
        p[-1] += diff
        
        out = ' '.join(f"{D1:.{decimal_places}f} {P1:.4f}" for D1, P1 in zip(dia, p))
        out = ' '.join([f"{num_points}",out])
        fig, ax = plt.subplots(figsize=(8, 5))  
        ax.plot(dia, p)
        ax.set_xlabel("Diameter")
        ax.set_ylabel("Percentage")
        fig.savefig(os.path.join(self.workdir,"particle_grade.png"), dpi=300)
        plt.close(fig) 

        return out
    
    @staticmethod
    def set_outstep(x):
    #输出步设置
        x = int(x)
        if x < 10:
            return x
        digits = len(str(x))  # 计算 x 的位数
        scale_factor = 10 ** (digits - 2)  # 计算缩放因子
        return (x // scale_factor) * scale_factor  # 取前两位，后面补零

    @staticmethod
    def set_restartstep(x, target_step):
    #重启动步数设置
        x = int(x)
        for out_down in range(x, 0, -1):
            if target_step % out_down == 0:
                return out_down
        for out_up in range(x, target_step + 1):
            if target_step % out_up == 0:
                return out_up

    @staticmethod
    def set_runstep(x):
    #运行步数设置
        x = int(x)
        factor = 10 ** (math.floor(math.log10(x)) - 1)  # 计算缩放因子
        return math.ceil(x / factor) * factor
    
    def save_load_conditions(self, load_con, filedir):
    # 储存加载条件
        df = pd.DataFrame(load_con, index=[0])
        df.to_csv(os.path.join(filedir, self.load_con_filename), index=False, header=True)
    
# #     def load_plate_data(self, workdir, reload=False):
# #         """
# #         加载边界板数据
# #         """
# #         cache_filepath = os.path.join(workdir, "plate_raw.txt")
        
# #         if not reload and os.path.exists(cache_filepath):
# #             return pd.read_csv(cache_filepath)
        
# #         num_of_headline = 2
# #         filepath = os.path.join(workdir, "plate.txt")
        
# #         try: 
# #             with open(filepath, 'r') as f:
# #                 lines = f.readlines()
# #             header_line = lines[1].lstrip('#').strip()  
# #             header = header_line.split() 
# #             data = pd.read_csv(filepath, sep=' ', skiprows=num_of_headline, header=None, names=header)
# #         except FileNotFoundError:
# #             print(f"Error: File '{filepath}' not found.")
# #             return None
    
# #         out = pd.DataFrame({
# #             "step": data['TimeStep'],
# #             "t": data['v_m_time'],
# #             "ss": abs(data['f_brlc[1]'] * self.pcc),
# #             "ns": abs(data['f_brlc[3]'] * self.pcc),
# #             "thk": data['c_2[3]'] - data['c_1[3]'],
# #         })
# #         out.to_csv(cache_filepath, index=False)
# #         return out
    
# #     @staticmethod
# #     def show_all(workdir,data):
# #         resultdir = os.path.join(workdir,r"raw_plot")
# #         os.makedirs(resultdir,exist_ok=True)
        
# #         for idx, key in enumerate(data.keys()):
# #             fig, ax = plt.subplots()
# #             ax.plot(data['t'], data[key], 'b')
# #             ax.set_ylabel(key)
# #             ax.set_xlabel('time/s')
# #             plt.tight_layout()
# #             fig.savefig(os.path.join(resultdir,f'{key}.png'), format='png', dpi=300)
# #             plt.close(fig)

    def check_is_completed(self, workdir, datadir, target_step):
        """检查当前工况是否完成"""
        # 首先检查工作目录
        flag = 1
        target_file = os.path.join(workdir, "restart", f"restart{target_step:d}.bishear")
        if not os.path.exists(target_file):
            flag = 0
        
            # 检查重启动文件目录
            load_con_dir = os.path.basename(workdir)
            sample_dir = os.path.basename(os.path.dirname(os.path.dirname(workdir)))
            rel_path = os.path.join(sample_dir,load_con_dir)
            restartfilepath = os.path.join(datadir, rel_path, "restart", f"restart{target_step}.bishear")
            if os.path.exists(restartfilepath):
                os.makedirs(load_con_dir, exist_ok=True)
                shutil.copytree(os.path.join(datadir, rel_path), load_con_dir, dirs_exist_ok=True)
                flag = 1
                
        return flag

    def load_restartfile(self,workdir, datadir, restart_step):
        """加载重启动文件"""
        load_con_dir = os.path.basename(workdir)
        last_load_con_dir = self.get_last_load_con_dir(load_con_dir)
        
        sample_dir = os.path.basename(os.path.dirname(workdir))

        rel_path = os.path.join(sample_dir,last_load_con_dir)
        
        restartfilepath = os.path.join(datadir, rel_path, "restart", f"restart{restart_step}.bishear")
        target_dir = os.path.join(workdir, "restart")
    
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
    
        if os.path.exists(restartfilepath):
            shutil.copy(restartfilepath, target_dir)
        else:
            print(f"in {workdir}, 重启动文件加载失败")
    
    @staticmethod
    def get_last_load_con_dir(workdir):
        """
        接受类似 "icon_5.0_ivel_0.1_" 的字符串，
        先使用 '_' 分割字符串，
        然后删除最后两个部分，
        最后用 '_' 拼接返回结果（末尾保留一个 '_'）
        """
        # 去掉末尾多余的下划线，避免产生空字段
        trimmed = workdir.rstrip('_')
        parts = trimmed.split('_')
        
        #处理施加围压的前序阶段
        if len(parts) == 2:
            return "gen"
        # 删除最后两个字段
        new_parts = parts[:-2]
        # 用下划线重新拼接，并在末尾添加下划线
        return '_'.join(new_parts) + '_'
        
    @staticmethod
    def check_restart_step(workdir,target_step,restart_step):
        """检查重启动步数设置"""
        flag = 1
        if (target_step % restart_step) !=0:
            print(f"in {workdir}，重启动步数设置错误 ")
            flag = 0
        return flag
    
    def save_load_con_file(self, workdir, load_con):
        """储存加载工况文件"""
        df = pd.DataFrame(load_con, index=[0])
        df.to_csv(os.path.join(workdir,self.load_con_filename),index=False, header=True)
        
    @staticmethod
    def save_restartfile(workdir, datadir, target_step):
        """储存重启动文件"""
        source_file = os.path.join(workdir, "restart", f"restart{target_step:d}.bishear")
        
        # 解析 workdir 以获取相对路径部分
        rel_path = os.path.relpath(workdir, os.path.dirname(os.path.dirname(os.path.abspath(workdir))))
        target_dir = os.path.join(datadir, rel_path, "restart")
        
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
    
        # 复制文件
        if os.path.exists(source_file):
            shutil.copy(source_file, target_dir)
        else:
            print(f"{source_file} 不存在，无法储存")

class GenPackingSimulation(BaseSimulation):
    # time scale sqrt(2*0.04/g) ~ 0.03s
    # gen_time = 0.1s ~  3t_g
    # 重力放大10倍 98.1m/s2
    gen_step = 100000
    gen_timestep = 5e-7
    
    
    def __init__(self, workdir, params = dict(), datadir = "restart_data", norun=False):
        super().__init__(workdir, params, datadir, norun)
        
        self.set_p_grade()
    
    def set_p_grade(self):
        self.p_grade = self.particle_grade_normal(0.003,0.001,0.001,0.005,10)
        pass

    def gen_packing(self):
        #文件路径设置
        workdir = os.path.join(self.workdir,"gen")
        
        input_filename = self.gen_filename

        # 载入参数
        seed = self.params['seed']
        lenx = self.params['lenx']
        leny = self.params['leny']
        kn   = self.params['kn']
        kt   = self.params['kt']
        gamman = self.params['gamman']
        gammat = self.params['gammat']
        xmu  = self.params['xmu']
        kroll = self.params['kroll']
        gammaroll = self.params['gammaroll']
        muroll = self.params['muroll']
        seed = self.params['seed']
         
        # 本地参数
        p_grade = self.p_grade
        gen_step = self.gen_step
        gen_timestep = self.gen_timestep
        run_step = gen_step
        target_step = gen_step
        
        #输出设置
        print_step = self.set_outstep(run_step/100)
        dump_step = self.set_outstep(run_step/2)
        restart_step = self.set_restartstep(run_step/1,target_step)
        
        if self.check_is_completed(workdir, self.datadir, target_step):
            return 
        
        self.check_restart_step(workdir, target_step, restart_step)
        
        #生成输入文件
        os.makedirs(workdir,exist_ok = True)
        with open(os.path.join(workdir,input_filename), 'w') as file:
            file.write(f"""
#! bin/bash
# file header
atom_style  sphere
atom_modify	map array
dimension   3

boundary	 p p f
newton		 off
comm_modify	 vel yes
units		 si
region		 reg block 0 {lenx} 0 {leny} -0.05 0.046 units box
create_box	 2 reg
neighbor	 0.001 bin
neigh_modify every 1 delay 0

pair_style granular
pair_coeff * * hertz {kn} {gamman} tangential linear_history {kt} {gammat} {xmu}

region          bc_t block 0 {lenx} 0 {leny}  0.0360 0.0405 units box
lattice		    sc 0.005
create_atoms	2 region bc_t
group		    tlattice region bc_t
set		        group tlattice density 2500 diameter 0.005
velocity        tlattice zero linear
velocity        tlattice zero angular
fix		        trlc tlattice aveforce NULL NULL NULL

region          bc_b block 0 {lenx} 0 {leny} -0.0082 -0.0042 units box
lattice         sc 0.005
create_atoms    2 region bc_b
group           blattice region bc_b
set             group blattice density 2500 diameter 0.005
velocity        blattice zero linear
velocity        blattice zero angular
fix             brlc blattice aveforce NULL NULL NULL
fix             rigid_ltb all rigid group 2 tlattice blattice force 1 off off off torque 1 off off off force 2 off off off torque 2 off off off

timestep {gen_timestep}

fix gravi all gravity 98.1 vector 0.0 0.0 -1.0

region    region_gouge_0 block 0 {lenx} 0 {leny} -0.0025 0.000 units box
region    region_gouge_1 block 0 {lenx} 0 {leny} 0.000 0.004 units box
region    region_gouge_2 block 0 {lenx} 0 {leny} 0.004 0.008 units box
region    region_gouge_3 block 0 {lenx} 0 {leny} 0.008 0.012 units box
region    region_gouge_4 block 0 {lenx} 0 {leny} 0.012 0.016 units box
region    region_gouge_5 block 0 {lenx} 0 {leny} 0.016 0.020 units box
region    region_gouge_6 block 0 {lenx} 0 {leny} 0.020 0.024 units box
region    region_gouge_7 block 0 {lenx} 0 {leny} 0.024 0.028 units box
region    region_gouge_8 block 0 {lenx} 0 {leny} 0.028 0.032 units box
region    region_gouge_9 block 0 {lenx} 0 {leny} 0.032 0.0375 units box

group nve_group region region_gouge_0
group nve_group region region_gouge_1
group nve_group region region_gouge_2
group nve_group region region_gouge_3
group nve_group region region_gouge_4
group nve_group region region_gouge_5
group nve_group region region_gouge_6
group nve_group region region_gouge_7
group nve_group region region_gouge_8
group nve_group region region_gouge_9

fix ins_0 nve_group pour 50000 1 {seed} region region_gouge_0 vol 0.5 200 diam poly {p_grade}
fix ins_1 nve_group pour 50000 1 {seed} region region_gouge_1 vol 0.5 200 diam poly {p_grade}
fix ins_2 nve_group pour 50000 1 {seed} region region_gouge_2 vol 0.5 200 diam poly {p_grade}
fix ins_3 nve_group pour 50000 1 {seed} region region_gouge_3 vol 0.5 200 diam poly {p_grade}
fix ins_4 nve_group pour 50000 1 {seed} region region_gouge_4 vol 0.5 200 diam poly {p_grade}
fix ins_5 nve_group pour 50000 1 {seed} region region_gouge_5 vol 0.5 200 diam poly {p_grade}
fix ins_6 nve_group pour 50000 1 {seed} region region_gouge_6 vol 0.5 200 diam poly {p_grade}
fix ins_7 nve_group pour 50000 1 {seed} region region_gouge_7 vol 0.5 200 diam poly {p_grade}
fix ins_8 nve_group pour 50000 1 {seed} region region_gouge_8 vol 0.5 200 diam poly {p_grade}
fix ins_9 nve_group pour 50000 1 {seed} region region_gouge_9 vol 0.5 200 diam poly {p_grade}

set group nve_group density 2500

fix integr nve_group nve/sphere

compute 1 blattice com
compute 2 tlattice com

#厚度
variable thk equal c_2[3]-c_1[3]

#体积分数
variable vol_atom atom 4/3*3.14*radius^3
compute vol_total nve_group reduce sum v_vol_atom
variable vol_frac equal c_vol_total/({lenx}*{leny}*0.04)

thermo_style custom step atoms f_brlc[1] f_brlc[3] c_1[1] c_1[3]
thermo {print_step}
thermo_modify lost ignore norm no

shell mkdir post
shell mkdir restart

variable m_time equal time
variable m_atoms equal atoms

fix ave_data all ave/time 2 10 {print_step} v_m_time v_m_atoms f_brlc[1] f_brlc[3] c_1[1] v_thk v_vol_frac file plate.txt title1 \"\" title2 "step time atoms sf nf lx thk vol"

dump dmp all cfg {dump_step} post/dump*.cfg mass type xs ys zs id x y z vx vy vz fx fy fz radius

run 1

restart {restart_step} restart/restart*.bishear

run {run_step} upto 
            """)
        if not self.norun:
            lmp_run_command(os.path.join(workdir,input_filename))
            
        #储存数据文件
        self.save_restartfile(workdir,self.datadir,target_step)
        
    def run(self):
        self.gen_packing()

class ConfiningSimulation(GenPackingSimulation):
    # time scale D * sqrt(rho/P) ~ 0.009s
    # con_time = 0.02s ~ 20 t_i
    con_step = 100000
    con_timestep = 1e-6
    load_time = 1e-4
    load_step = int(load_time/con_timestep)
    
    def __init__(self, workdir, params = dict(), datadir = "restart_data", norun=False):
        super().__init__(workdir, params, datadir, norun)

    def confining(self, load_con):
        #文件路径设置
        icon = load_con['icon']
        workdir = os.path.join(self.workdir,f"icon_{icon}_")
        
        input_filename = self.con_filename

        # 载入参数
        lenx = self.params['lenx']
        leny = self.params['leny']
        kn   = self.params['kn']
        kt   = self.params['kt']
        gamman = self.params['gamman']
        gammat = self.params['gammat']
        xmu  = self.params['xmu']
        
        # 本地参数
        gen_step = self.gen_step
        con_step = self.con_step
        timestep = self.con_timestep
        run_step = con_step
        last_step = gen_step
        target_step = gen_step + con_step
        
        #输出设置
        print_step = self.set_outstep(run_step/100)
        dump_step = self.set_outstep(run_step/2)
        restart_step = self.set_restartstep(run_step/5,target_step)
        
        if self.check_is_completed(workdir, self.datadir, target_step):
            return 
        
        self.load_restartfile(workdir, self.datadir, last_step)
        
        self.check_restart_step(workdir, target_step, restart_step)
        
        self.save_load_con_file(workdir, load_con)
        
        #定义线性加载
        #注意是每个原子加的力
        load_step = self.load_step
        start_step = last_step
        end_step = start_step + load_step
        start_force = 0
        target_force = icon*1e6*lenx*leny/144
        
        #定义执行时间步
        run_step1 = load_step
        run_step2 = run_step - run_step1
        
        #生成输入文件
        os.makedirs(workdir,exist_ok = True)
        with open(os.path.join(workdir,input_filename), 'w') as file:
            file.write(f"""
#! bin/bash
# file header
atom_style  sphere
atom_modify	map array
dimension   3
boundary	p p f
newton		off
comm_modify	vel yes
units		si

read_restart restart\\restart{last_step}.bishear

region		 reg block 0 {lenx} 0 {leny} 0.0360 0.0405 units box
neighbor	 0.001 bin
neigh_modify every 1 delay 0

pair_style granular
pair_coeff * * hertz {kn} {gamman} tangential linear_history {kt} {gammat} {xmu} 

group       tlattice id 1:144
set         group tlattice density 2500 diameter 0.005
velocity    tlattice zero linear
velocity    tlattice zero angular
fix		    trlc tlattice setforce 0 0 0

group       blattice id 145:288
set         group blattice density 2500 diameter 0.005
velocity    blattice zero linear
velocity    blattice zero angular
fix         brlc blattice aveforce NULL NULL NULL

fix rigid_ltb all rigid group 2 tlattice blattice force 1 off off off torque 1 off off off force 2 off off on torque 2 off off off

variable force_linear equal \"({start_force} + ({target_force}-{start_force}) * (step-{start_step}) / ({end_step}-{start_step}))\"
fix force_c_b blattice addforce 0.0 0.0 v_force_linear

timestep   {timestep}

fix gravi all gravity 9.81 vector 0.0 0.0 -1.0

group  nve_group id 289:40000
set    group nve_group density 2500
fix    integr nve_group nve/sphere

compute    1 blattice com
compute    2 tlattice com

#厚度
variable thk equal c_2[3]-c_1[3]

#体积分数
variable vol_atom atom 4/3*3.14*radius^3
compute vol_total nve_group reduce sum v_vol_atom
variable vol_frac equal c_vol_total/({lenx}*{leny}*v_thk)

thermo    500
thermo_modify    lost ignore norm no

shell    mkdir post
shell    mkdir restart

restart {restart_step} restart/restart*.bishear

variable m_time equal time
variable m_atoms equal atoms

fix ave_data all ave/time 2 10 {print_step} v_m_time v_m_atoms f_brlc[1] f_brlc[3] c_1[1] v_thk v_vol_frac file plate.txt title1 \"\" title2 "step time atoms sf nf lx thk vol"

dump dmp all cfg {dump_step} post/dump*.cfg mass type xs ys zs id x y z vx vy vz fx fy fz radius

run {run_step1}

variable force_linear equal {target_force}

run {run_step2}
            """)
        if not self.norun:
            lmp_run_command(os.path.join(workdir,input_filename))
            
        #储存数据文件
        self.save_restartfile(workdir,self.datadir,target_step)
        
    def run(self):
        GenPackingSimulation.run(self)
        # list(executor2.map(lambda load_con: self.confining(load_con), self.con_list))
        
        for load_con in self.con_list:
            self.confining(load_con)
        
class SlidingSimulation(ConfiningSimulation):
    slide_timestep = 1e-6
    
    def __init__(self, workdir, params = dict(), datadir = "restart_data", norun=False):
        super().__init__(workdir, params, datadir, norun)

    def sliding(self, load_con):
        #文件路径设置
        icon = load_con['icon']
        ivel = load_con['ivel']
        workdir = os.path.join(self.workdir,f"icon_{icon}_ivel_{ivel}_")
        
        input_filename = self.slide_filename

        # 载入参数
        lenx = self.params['lenx']
        leny = self.params['leny']
        kn   = self.params['kn']
        kt   = self.params['kt']
        gamman = self.params['gamman']
        gammat = self.params['gammat']
        xmu  = self.params['xmu']
        k_s = self.params['k_s']
        
        # 本地参数
        gen_step = self.gen_step
        con_step = self.con_step
        timestep = self.slide_timestep
        
        ss_step  = self.set_runstep(10 * 0.003 / ivel / timestep) # 10个颗粒直径 
        
        run_step = ss_step
        last_step = gen_step + con_step
        target_step = gen_step + con_step + ss_step
        
        #输出设置
        run_step2 = run_step//20 #精细化场输出
        run_step1 = run_step - run_step2
        
        print_step = self.set_outstep(0.001 * 0.003 / ivel / timestep) # 输出频率0.001Dmean 这里保持扰动后相同的输出频率方便数据处理

        dump_step1 = self.set_outstep(run_step/2)
        dump_step2 = self.set_outstep(0.1 * 0.003 / ivel / timestep) # 输出频率0.05Dmean 
        restart_step = self.set_restartstep(run_step/2,target_step) 
        
        if self.check_is_completed(workdir, self.datadir, target_step):
            return 
        
        self.load_restartfile(workdir, self.datadir, last_step)
        
        self.check_restart_step(workdir, target_step, restart_step)
        
        self.save_load_con_file(workdir, load_con)
        
        #生成输入文件
        os.makedirs(workdir,exist_ok = True)
        with open(os.path.join(workdir,input_filename), 'w') as file:
            file.write(f"""
#! bin/bash
# file header
atom_style  sphere
atom_modify	map array
dimension   3
boundary	p p f
newton		off
comm_modify	vel yes
units		si

read_restart restart\\restart{last_step}.bishear

region		 reg block 0 {lenx} 0 {leny} 0.0360 0.0405 units box
neighbor	 0.001 bin
neigh_modify every 1 delay 0

pair_style granular
pair_coeff * * hertz {kn} {gamman} tangential linear_history {kt} {gammat} {xmu} 

group       tlattice id 1:144
set         group tlattice density 2500 diameter 0.005
velocity    tlattice zero linear
velocity    tlattice zero angular
fix		    trlc tlattice setforce 0 0 0

group       blattice id 145:288
set         group blattice density 2500 diameter 0.005
velocity    blattice zero linear
velocity    blattice zero angular
fix         brlc blattice aveforce NULL NULL NULL

fix rigid_ltb all rigid group 2 tlattice blattice force 1 off off off torque 1 off off off force 2 on off on torque 2 off off off

fix force_c_b blattice smd cfor {-icon*1e6*lenx*leny} tether NULL NULL -0.05 0.0
fix pull blattice smd cvel {k_s} {ivel} tether {lenx} NULL NULL 0.0

timestep   {timestep}

fix gravi all gravity 9.81 vector 0.0 0.0 -1.0

group  nve_group id 289:40000
set    group nve_group density 2500
fix    integr nve_group nve/sphere

compute    1 blattice com
compute    2 tlattice com
compute    3 all property/local patom1 patom2
compute    4 all pair/local  dist fx fy fz p1 p2 p3

#厚度
variable thk equal c_2[3]-c_1[3]

#体积分数
variable vol_atom atom 4/3*3.14*radius^3
compute vol_total nve_group reduce sum v_vol_atom
variable vol_frac equal c_vol_total/({lenx}*{leny}*v_thk)

#配位数
compute contact_atom nve_group contact/atom
compute contact_total nve_group reduce sum c_contact_atom
variable coordination equal c_contact_total/(atoms-288)

#热力学输出
thermo_style    custom step atoms f_brlc[1] f_brlc[3] c_1[1] c_1[3]
thermo    {print_step}
thermo_modify    lost ignore norm no

shell    mkdir post
shell    mkdir restart
shell    mkdir pair

restart {restart_step} restart/restart*.bishear

variable m_time equal time
variable m_atoms equal atoms

fix ave_data all ave/time 2 10 {print_step} v_m_time v_m_atoms f_brlc[1] f_brlc[3] c_1[1] v_thk v_vol_frac file plate.txt title1 \"\" title2 "step time atoms sf nf lx thk vol"

dump dmp all cfg {dump_step1} post/dump*.cfg mass type xs ys zs id x y z vx vy vz fx fy fz radius
# id1 id2 dist fx fy fz tx ty tz
# dump dmp2 all local {dump_step1} pair/pair*.txt index c_3[1] c_3[2] c_4[1] c_4[2] c_4[3] c_4[4] c_4[5] c_4[6] c_4[7]

# 云图输出
# region chunk_region block 0 {lenx} 0 {leny} -0.0002 0.002 units box
# compute mychunk all chunk/atom bin/1d z 0.002 0.0002 units box region chunk_region #先测试10个分区
# fix cloud all ave/chunk 5 {dump_step1//10} {dump_step1} mychunk vx file cloud.txt #与接触对保持相同输出频率

run {run_step} 

#dump_modify dmp every {dump_step2}
#dump_modify dmp2 every {dump_step2}

#unfix cloud
#fix cloud2 all ave/chunk 5 {dump_step2//10} {dump_step2} mychunk vx append cloud2.txt #与接触对保持相同输出频率

#run {run_step2} 
            """)
        if not self.norun:
            lmp_run_command(os.path.join(workdir,input_filename))
            
        #储存数据文件
        self.save_restartfile(workdir,self.datadir,target_step)
        
        #calibration接口
        shutil.copy(os.path.join(workdir,"plate.txt"), os.path.dirname(os.path.dirname(self.workdir)))
        
    def run(self):
        ConfiningSimulation.run(self)
        # list(executor2.map(lambda load_con: self.sliding(load_con), self.slide_list))
        for load_con in self.slide_list:
            self.sliding(load_con)
        
class NssSimulation(SlidingSimulation):
    nss_timestep = 5e-9
    
    def __init__(self, workdir, params = dict(), datadir = "restart_data", norun=False):
        super().__init__(workdir, params, datadir, norun)

    def nss(self, load_con):
        #文件路径设置
        icon = load_con['icon']
        ivel = load_con['ivel']
        fcon = load_con['fcon']
        fcon_value = icon * (1+fcon)
        workdir = os.path.join(self.workdir,f"icon_{icon}_ivel_{ivel}_fcon_{fcon}")
        
        input_filename = self.slide_filename

        # 载入参数
        lenx = self.params['lenx']
        leny = self.params['leny']
        kn   = self.params['kn']
        kt   = self.params['kt']
        gamman = self.params['gamman']
        gammat = self.params['gammat']
        xmu  = self.params['xmu']
        kroll = self.params['kroll']
        gammaroll = self.params['gammaroll']
        muroll = self.params['muroll']
        k_s = self.params['k_s']
        
        # 本地参数
        gen_step = self.gen_step
        con_step = self.con_step
        ss_timestep = self.ss_timestep
        ss_step  = self.set_runstep(10 * 0.003 / ivel / ss_timestep) # 10个颗粒直径 

        timestep = self.nss_timestep
        nss_step = self.set_runstep(6 * 0.003 / ivel / timestep) # 6个颗粒直径
        
        run_step = nss_step
        last_step = gen_step + con_step + ss_step
        target_step = last_step + nss_step
        
        #输出设置
        run_step1 = run_step//10 #精细化场输出
        run_step2 = run_step - run_step1
        
        print_step = self.set_outstep(0.0001 * 0.003 / ivel / timestep) # 输出频率0.0001Dmean 这里保持扰动后相同的输出频率方便数据处理

        dump_step2 = self.set_outstep(run_step/10)
        dump_step1 = self.set_outstep(0.05 * 0.003 / ivel / timestep) # 输出频率0.05Dmean 
        restart_step = self.set_restartstep(run_step/10,target_step) 
        
        if self.check_is_completed(workdir, target_step):
            return 
        
        self.load_restartfile(workdir, self.datadir, last_step)
        
        self.check_restart_step(workdir, target_step, restart_step)
        
        self.save_load_con_file(workdir, load_con)
        
        #定义线性加载
        #注意是每个原子加的力
        start_step = last_step
        load_step = int(0.01 * 0.003 / ivel / timestep) #0.01Dmean
        end_step = start_step +load_step
        start_force = icon*1e6*lenx*leny/400
        target_force = fcon_value*1e6*lenx*leny/400
        
        #生成输入文件
        os.makedirs(workdir,exist_ok = True)
        with open(os.path.join(workdir,input_filename), 'w') as file:
            file.write(f"""
#! bin/bash
# file header
atom_style  sphere
atom_modify	map array
dimension   3
boundary	p p f
newton		off
comm_modify	vel yes
units		si

read_restart restart\\restart{last_step}.bishear

region		 reg block 0 {lenx} 0 {leny} 0.0360 0.0405 units box
neighbor	 0.001 bin
neigh_modify every 1 delay 0

pair_style granular
pair_coeff * * hertz {kn} {gamman} tangential linear_history {kt} {gammat} {xmu} rolling sds {kroll} {gammaroll} {muroll}

group       tlattice id 1:144
set         group tlattice density 2500 diameter 0.005
velocity    tlattice zero linear
velocity    tlattice zero angular
fix		    trlc tlattice setforce 0 0 0

group       blattice id 145:288
set         group blattice density 2500 diameter 0.005
velocity    blattice zero linear
velocity    blattice zero angular
fix         brlc blattice aveforce NULL NULL NULL

fix rigid_ltb all rigid group 2 tlattice blattice force 1 off off off torque 1 off off off force 2 on off on torque 2 off off off

#定义线性加载
variable force_linear equal \"(step <= {end_step}) ? {start_force} + ({target_force} - {start_force})*(step - {start_step})/({end_step} - {start_step}) : {target_force}\"
fix force_c_b blattice addforce 0.0 0.0 v_force_linear

#施加水平剪切速度
fix pull blattice smd cvel {k_s} {ivel} tether {lenx} NULL NULL 0.0

timestep   {timestep}

fix gravi all gravity 9.81 vector 0.0 0.0 -1.0

group  nve_group id 800:40000
set    group nve_group density 2500
fix    integr nve_group nve/sphere

compute    1 blattice com
compute    2 tlattice com
compute    3 all property/local patom1 patom2
compute    4 all pair/local  dist fx fy fz p1 p2 p3

#厚度
variable thk equal c_2[3]-c_1[3]

#体积分数
variable vol_atom atom 4/3*3.14*radius^3
compute vol_total nve_group reduce sum v_vol_atom
variable vol_frac equal c_vol_total/({lenx}*{leny}*v_thk)

#配位数
compute contact_atom nve_group contact/atom
compute contact_total nve_group reduce sum c_contact_atom
variable coordination equal c_contact_total/(atoms-800)

#热力学输出
thermo_style    custom step atoms f_brlc[1] f_brlc[3] c_1[1] c_1[3]
thermo    {print_step}
thermo_modify    lost ignore norm no

shell    mkdir post
shell    mkdir restart
shell    mkdir pair

restart {restart_step} restart/restart*.bishear

variable m_time equal time
variable m_atoms equal atoms

fix ave_data all ave/time 10 20 {print_step} v_m_time v_m_atoms f_brlc[1] f_brlc[3] c_1[1] v_thk v_vol_frac file plate.txt title1 \"\" title2 "step time atoms sf nf lx thk vol"

dump dmp all cfg {dump_step1} post/dump*.cfg mass type xs ys zs id x y z vx vy vz fx fy fz radius
# id1 id2 dist fx fy fz tx ty tz
dump dmp2 all local {dump_step1} pair/pair*.txt index c_3[1] c_3[2] c_4[1] c_4[2] c_4[3] c_4[4] c_4[5] c_4[6] c_4[7]

# 云图输出
region chunk_region block 0 {lenx} 0 {leny} -0.0002 0.002 units box
compute mychunk all chunk/atom bin/1d z 0.002 0.0002 units box region chunk_region #先测试10个分区
fix cloud all ave/chunk 5 {dump_step1//10} {dump_step1} mychunk vx file cloud.txt #与接触对保持相同输出频率

run {run_step1} 

dump_modify dmp every {dump_step2}
dump_modify dmp2 every {dump_step2}

# unfix cloud
# fix cloud all ave/chunk 5 {dump_step2//10} {dump_step2} mychunk vx append cloud.txt #与接触对保持相同输出频率

run {run_step2} 
            """)
        if not self.norun:
            lmp_run_command(os.path.join(workdir,input_filename))
            
        #储存数据文件
        self.save_restartfile(workdir,self.datadir,target_step)
        
    def run(self):
        SlidingSimulation.run(self)
        list(executor2.map(lambda load_con: self.nss(load_con), self.nss_list))

def submit_one_sample(workdir, params = dict(), datadir = "restart_data", norun=False):
    # GenPackingSimulation(workdir, params, datadir, norun).run()
    # ConfiningSimulation(workdir, params, datadir, norun).run()
    SlidingSimulation(workdir, params, datadir, norun).run()
    # NssSimulation(workdir, params, datadir, norun).run()
# 
def run_nss_simulation(workdir, params = dict(), datadir = "restart_data", norun=False):
    icon_list = [0.01]
    # fcon_list = [0.2]
    ivel_list = [0.1]
    seeds = ["14499",]
    
    params_list = [{"seed": seed} for seed in seeds]
    
    #分别提交每个加载速度下的运行工况
    for ivel in ivel_list:
        load_conditions = {
            "icon": icon_list,
            "ivel": [ivel,],
            # "fcon": fcon_list,
            }
        for params in params_list:
            params["load_conditions"] = load_conditions
        
        #
        # list(executor1.map(lambda params: submit_one_sample(workdir, params, datadir, norun), params_list))
        
        for params in params_list:
            submit_one_sample(workdir, params, datadir, norun)

# calibration接口  
def run_simulation(workdir, params):
    # kn gamma xmu
    icon_list = [0.01]
    ivel_list = [0.1]
    params['kt'] = params['kn'] * 3 * 0.7 /1.7
    params['gammat'] = params['gamman'] * 0.5
    for ivel in ivel_list:
        load_conditions = {
            "icon": icon_list,
            "ivel": [ivel,],
            # "fcon": fcon_list,
            }
        params["load_conditions"] = load_conditions
        SlidingSimulation(workdir, params).run()
    
def get_results(workdir):
    import data_proc as dp

    try:
        file = os.path.join(workdir,"plate.txt")
        data =pd.read_csv(file,sep =" ")
        # atoms = data['atoms']
        ss = np.abs(data['sf']/0.06/0.06/1000) #kPa
        ns = np.abs(data['nf']/0.06/0.06/1000) #kPa
        # thk = data['thk']
        
        lx = data['lx']
        lx = dp.zero(lx)
        
        window = int(len(lx[lx<0.003]) * 0.02) 
        #读取刚度
        slopes = dp.rslope(lx, ss*1e3, window=window)  # 计算局部斜率，单位 Pa/m
        k_load = np.max(slopes)
        
        #读取峰值强度
        ss_peak  = np.max(ss)
        
        #读取稳定摩擦系数
        mu = dp.divide(ss,ns,window) 
        condition = lx[-1]-lx<5*0.003
        mu_mean = np.mean(mu[condition])

        return [k_load, ss_peak, mu_mean]

    except Exception as e:
        print(f"⚠️ 读取 {file} 失败: {e}")
        return [np.nan, np.nan, np.nan]

if __name__ == "__main__":
    # 标准模型工作流
    workdir = os.getcwd()
    run_nss_simulation(workdir)

    # 测试代码
    # workdir = os.getcwd()
    # ConfiningSimulation(workdir).run()


