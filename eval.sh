'''
--model_type 参数用于选择你要训练或评估的神经网络模型架构。在本项目中，它有两个可选值，分别对应两种不同的假设模型：
    pc (Predictive Coding Network)
        对应类：PredictiveCodingRNN
        原理：这是本项目的核心模型。它显式地模拟了“预测编码”机制。网络分为两层，底层计算预测误差 (Prediction Error)，顶层维护内部状态。
        用途：用来验证你的核心假设——即“预测误差”信号是否能解释 MMN (Mismatch Negativity) 现象。它的输出包含显式的误差信号et。
    baseline (Baseline RNN)
        对应类：BaselineRNN
        原理：这是一个普通的标准循环神经网络 (GRU)。它只负责根据当前输入预测下一个 token，内部没有专门计算“误差”的神经元模块。
        用途：作为对照组。用来证明 MMN 现象不仅仅是普通神经网络的自然属性，而是特定结构（预测编码结构）带来的效应。
        在评估时，我们用它的 Loss 大小来强行模拟“惊奇度”，对比它与 PC 模型的反应差异。
'''
python src/train.py --model_type baseline --epochs 20 --p_oddball 0.1 --output_dir outputs
python src/eval.py --model_type baseline --p_oddball 0.1 --output_dir outputs