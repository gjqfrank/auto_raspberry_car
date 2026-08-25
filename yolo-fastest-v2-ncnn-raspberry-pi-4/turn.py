import pigpio
import time

# ====================== 四个电机引脚定义（和C++完全一样）======================
LF_FWD = 1    # 左前前进
LF_BWD = 7    # 左前后退
LF_PWM = 12   # 左前调速

LB_FWD = 24   # 左后前进
LB_BWD = 23   # 左后后退
LB_PWM = 18   # 左后调速

RF_FWD = 6    # 右前前进
RF_BWD = 5    # 右前后退
RF_PWM = 13   # 右前调速

RB_FWD = 21   # 右后前进
RB_BWD = 20   # 右后后退
RB_PWM = 19   # 右后调速

# ====================== PWM 参数 ======================
freq_ch0 = 100        # PWM 频率
left_duty = 500000    # 左轮占空比（你C++里写的）
right_duty = 330000   # 右轮占空比（你C++里写的）
run_time = 3          # 运行 3 秒停止

# ====================== 主程序 ======================
print("Python pigpio PWM 测试启动")

# 连接 pigpio 守护进程
pi = pigpio.pi()
if not pi.connected:
    print("错误：无法连接到 pigpiod 服务！")
    exit()

# 设置所有引脚为输出模式
pins = [LF_FWD, LF_BWD, LF_PWM,
        LB_FWD, LB_BWD, LB_PWM,
        RF_FWD, RF_BWD, RF_PWM,
        RB_FWD, RB_BWD, RB_PWM]

for pin in pins:
    pi.set_mode(pin, pigpio.OUTPUT)

# ====================== 设置全部电机前进 ======================
# 左前
pi.write(LF_FWD, 1)
pi.write(LF_BWD, 0)

# 左后
pi.write(LB_FWD, 1)
pi.write(LB_BWD, 0)

# 右前
pi.write(RF_FWD, 1)
pi.write(RF_BWD, 0)

# 右后
pi.write(RB_FWD, 1)
pi.write(RB_BWD, 0)

# ====================== 启动 PWM 转速差 ======================
pi.hardware_PWM(LF_PWM, freq_ch0, left_duty)
pi.hardware_PWM(LB_PWM, freq_ch0, left_duty)
pi.hardware_PWM(RF_PWM, freq_ch0, right_duty)
pi.hardware_PWM(RB_PWM, freq_ch0, right_duty)

print("小车运行中... 运行 3 秒后停止")
time.sleep(run_time)  # 保持运行 3 秒

# ====================== 停止所有电机 ======================
# 停方向引脚
pi.write(LF_FWD, 0)
pi.write(LF_BWD, 0)
pi.write(LB_FWD, 0)
pi.write(LB_BWD, 0)
pi.write(RF_FWD, 0)
pi.write(RF_BWD, 0)
pi.write(RB_FWD, 0)
pi.write(RB_BWD, 0)

# 停 PWM
pi.hardware_PWM(LF_PWM, 0, 0)
pi.hardware_PWM(LB_PWM, 0, 0)
pi.hardware_PWM(RF_PWM, 0, 0)
pi.hardware_PWM(RB_PWM, 0, 0)

print("停止 PWM，程序结束")

# 断开连接
pi.stop()