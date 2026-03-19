# CARLA LSTM Predictive Driving

这个项目会做两件事：

1. 从 `dataset/` 里的 CSV 时序数据训练一个 LSTM 驾驶模型。
2. 在 CARLA 中生成一辆车，接受短暂 `WASD` 输入，然后把后续驾驶交给模型预测控制。

## 已实现内容

- `train_lstm.py`
  - 读取 `dataset/**/*.csv`
  - 构造历史窗口 `history_steps` 到未来预测窗口 `prediction_steps`
  - 训练 LSTM 输出未来油门、转向和局部轨迹
  - 保存模型与归一化参数到 `artifacts/models/`
- `run_carla_controller.py`
  - 连接 CARLA Server
  - 生成一辆车和 IMU 传感器
  - 监听 `WASD`
  - 你短暂操作后，模型预测接下来一段控制序列并自动执行
  - 用 CARLA debug 线条画出预测轨迹
- `main.py`
  - 统一入口
- `setup_env.ps1`
  - 自动配置虚拟环境并安装依赖

## 环境准备

当前项目按 Python 3.12 配置。

```powershell
.\setup_env.ps1
```

## 训练模型

```powershell
.\.venv\Scripts\python.exe main.py train
```

常用参数：

```powershell
.\.venv\Scripts\python.exe main.py train -- --epochs 40 --batch-size 256 --history-steps 20 --prediction-steps 15
```

训练输出：

- `artifacts/models/driving_lstm.pt`
- `artifacts/models/driving_lstm_metadata.json`
- `artifacts/plots/training_loss.png`

## 运行 CARLA 控制脚本

先启动 CARLA Server，再执行：

```powershell
.\.venv\Scripts\python.exe main.py run
```

可选参数：

```powershell
.\.venv\Scripts\python.exe main.py run -- --host 127.0.0.1 --port 2000 --spawn-index 0 --vehicle-filter vehicle.tesla.model3
```

## 控制方式

- `W`: 前进油门
- `S`: 刹车
- `A` / `D`: 左右转向
- 松开按键后：
  - 脚本会根据你最近一小段操作历史做 LSTM 预测
  - 自动执行预测出来的控制序列
  - 车辆按预测轨迹继续跑
- `R`: 清空当前预测
- `Esc`: 退出

## 说明

- 训练时使用的特征来自数据集里的控制、速度、IMU、位置和朝向。
- 运行时会从 CARLA 车辆状态和 IMU 传感器重建相同结构的输入特征。
- 由于这里无法直接替你启动 CARLA 图形端，联调需要你本机启动 CARLA 后再运行控制脚本。
