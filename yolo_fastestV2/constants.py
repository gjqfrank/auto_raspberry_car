"""
Constants and configuration file for Auto Raspberry Car Control
所有可调参数集中定义在这个文件中
"""

import numpy as np

# ============================================================================
# 网络配置 (Network Configuration)
# ============================================================================
STREAM_URL = "http://172.20.10.2:8080/?action=stream"      # 树莓派摄像头流地址
CONTROL_URL = "http://172.20.10.2:5000/control"            # 树莓派车辆控制地址

# ============================================================================
# 模型配置 (Model Configuration)
# ============================================================================
CONFIG_FILE = 'data/coco.data'                              # 模型配置文件
WEIGHTS_PATH = 'modelzoo/coco2017-0.241078ap-model.pth'    # 模型权重文件
DEVICE = "cpu"                                              # 运行设备 ("cpu" 或 "cuda")

# ============================================================================
# 推理参数 (Inference Parameters)
# ============================================================================
NMS_CONF_THRESHOLD = 0.3                                    # 置信度阈值
NMS_IOU_THRESHOLD = 0.4                                     # IoU 阈值

# ============================================================================
# 行人追踪参数 (Person Tracking Parameters)
# ============================================================================
PERSON_STOP_AREA_THRESHOLD = 0.7                            # 行人检测框面积阈值（停止距离）
PERSON_LEFT_THRESHOLD = 0.3                                 # 左转判断阈值
PERSON_RIGHT_THRESHOLD = 0.7                                # 右转判断阈值

# ============================================================================
# 车道检测参数 (Lane Detection Parameters)
# ============================================================================
# 白线颜色范围 (HSV)
LANE_LOWER_WHITE = np.array([0, 0, 100])
LANE_UPPER_WHITE = np.array([180, 30, 255])

# 车道曲率判断阈值
LANE_SHARP_LEFT_THRESHOLD = -0.5                            # 急左转阈值
LANE_GENTLE_LEFT_THRESHOLD = -0.15                          # 缓左转阈值
LANE_GENTLE_RIGHT_THRESHOLD = 0.15                          # 缓右转阈值
LANE_SHARP_RIGHT_THRESHOLD = 0.5                            # 急右转阈值

# 车道检测形态学操作参数
LANE_KERNEL_SIZE = (5, 5)                                   # 形态学操作的卷积核大小
LANE_ROI_START_RATIO = 0.5                                  # ROI 起始位置比例（从上到下的比例）

# ============================================================================
# 行人安全距离参数 (Person Safety Distance Parameters)
# ============================================================================
DANGER_DISTANCE_THRESHOLD = 0.50                            # 危险距离阈值 (50% 画面面积)
SAFE_DISTANCE_THRESHOLD = 0.35                              # 安全距离阈值 (35% 画面面积)
PERSON_SAFETY_MIN_COMMAND_INTERVAL = 0.05                   # 最小命令间隔(秒)
PERSON_SAFETY_DISTANCE_HISTORY_SIZE = 5                     # 距离历史缓冲大小

# ============================================================================
# 红绿灯检测参数 (Traffic Light Detection Parameters)
# ============================================================================
# 红色范围 (HSV) - 红色在 HSV 中分为两个范围
TRAFFIC_LIGHT_LOWER_RED1 = np.array([0, 100, 100])
TRAFFIC_LIGHT_UPPER_RED1 = np.array([10, 255, 255])
TRAFFIC_LIGHT_LOWER_RED2 = np.array([170, 100, 100])
TRAFFIC_LIGHT_UPPER_RED2 = np.array([180, 255, 255])

# 绿色范围 (HSV)
TRAFFIC_LIGHT_LOWER_GREEN = np.array([35, 100, 100])
TRAFFIC_LIGHT_UPPER_GREEN = np.array([85, 255, 255])

# 黄色范围 (HSV) - 可选
TRAFFIC_LIGHT_LOWER_YELLOW = np.array([15, 100, 100])
TRAFFIC_LIGHT_UPPER_YELLOW = np.array([35, 255, 255])

# 颜色检测阈值
TRAFFIC_LIGHT_RED_RATIO_THRESHOLD = 0.15                    # 红色像素占比阈值
TRAFFIC_LIGHT_GREEN_RATIO_THRESHOLD = 0.15                  # 绿色像素占比阈值
TRAFFIC_LIGHT_YELLOW_RATIO_THRESHOLD = 0.12                 # 黄色像素占比阈值

# 红绿灯类别 (Traffic Light Categories)
TRAFFIC_LIGHT_CATEGORIES = {"traffic light", "traffic_light", "light"}  # 检测模型中对应的类别名称

# ============================================================================
# 控制��令定义 (Control Commands)
# ============================================================================
COMMAND_FORWARD = "FORWARD"                                 # 前进
COMMAND_LEFT = "LEFT"                                       # 左转
COMMAND_RIGHT = "RIGHT"                                     # 右转
COMMAND_STOP = "STOP"                                       # 停止
COMMAND_CONTINUE = "CONTINUE"                               # 继续

# ============================================================================
# 调试和日志参数 (Debug and Logging Parameters)
# ============================================================================
DEBUG_MODE = True                                           # 调试模式开关
VERBOSE_LOGGING = True                                      # 详细日志开关
LOG_EVERY_N_FRAMES = 30                                     # 每 N 帧打印一次日志
LOG_INTERVAL = 30                                           # 日志输出间隔（帧数）

# ============================================================================
# 显示参数 (Display Parameters)
# ============================================================================
DISPLAY_LANE_MASK = True                                    # 显示车道掩码
DISPLAY_DETECTION_BOXES = True                              # 显示检测框
DISPLAY_MODE_INFO = True                                    # 显示模式信息
DISPLAY_STATS = True                                        # 显示统计信息
SHOW_VISUAL_OUTPUT = True                                   # 显示视觉输出

# ============================================================================
# 优先级定义 (Priority Levels)
# ============================================================================
"""
优先级系统 (Priority System):

Priority 1 (Highest): 🛡️  Person Safety Distance Detection
  - 行人距离太近时紧急停止
  
Priority 2 (High): 🚦 Traffic Light Detection
  - 检测红绿灯，遵守交通规则
  
Priority 3 (Medium): 👤 Person Detection
  - 检测到行人时停止
  
Priority 4 (Lowest): 🛣️  Lane Following
  - 默认的车道跟踪模式
"""

# ============================================================================
# 性能参数 (Performance Parameters)
# ============================================================================
THREAD_DAEMON_MODE = True                                   # 守护线程模式
CAPTURE_QUEUE_SIZE = 1                                      # 帧捕获队列大小
INFERENCE_TIMEOUT = 5.0                                     # 推理超时时间(秒)

# ============================================================================
# 环境参数 (Environment Parameters)
# ============================================================================
USE_GPU = False                                             # 是否使用 GPU
GPU_ID = 0                                                  # GPU 设备ID
NUM_WORKERS = 4                                             # 数据加载工作线程数

print("[CONFIG] Constants loaded successfully")
