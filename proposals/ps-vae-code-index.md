# PS-VAE项目完整代码索引方案

## 项目概述

**PS-VAE (Principal Subgraph Variational AutoEncoder)** 是一个基于主子图挖掘与组装的分子生成深度学习项目，发表于NeurIPS 2022。本方案提供项目的完整代码索引和技术架构指南，包含每个功能模块和函数的详细位置信息。

## 目标

1. 建立完整的代码结构索引
2. 提供模块功能映射和函数级别的详细定位
3. 创建技术依赖关系图
4. 制定代码导航指南
5. 建立开发和使用流程文档
6. 提供函数级别的API参考

## 代码架构索引

### 1. 核心模块 (`src/`)

#### 1.1 主要入口文件

**`src/train.py`** - 模型训练主程序 (160行)
- **功能**: PS-VAE模型训练流程控制
- **关键函数**: 
  - `setup_seed(seed)` - 第18行，随机种子设置，确保结果可复现
  - `train(model, train_loader, valid_loader, test_loader, args)` - 第42行，训练主循环
  - `parse()` - 第70行，命令行参数解析
- **依赖**: PyTorch Lightning框架，VAEEarlyStopping早停策略
- **配置**: 支持多GPU训练，梯度裁剪，模型检查点
- **训练策略**: KL散度预热，β退火机制

**`src/generate.py`** - 分子生成与属性优化 (362行)
- **功能**: 基于训练好的模型生成分子
- **关键函数**: 
  - `setup_seed(seed)` - 第15行，随机种子设置
  - `get_loss(model, props, target)` - 第22行，计算属性损失
  - `gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol=None)` - 第35行，单分子生成
  - `beam_gen(model, z, beam, target, max_atom_num, add_edge_th, temperature, constraint_mol=None)` - 第42行，集束搜索生成
  - `direct_optimize(model, z, target, lr, max_iter, patience, max_atom_num, add_edge_th, temperature)` - 第54行，直接梯度优化
  - `load_model(ckpt, gpus)` - 第79行，模型加载
- **支持属性**: QED、LogP、SA等分子性质优化
- **优化方法**: 梯度上升、约束优化、集束搜索

#### 1.2 核心神经网络模块 (`src/modules/`)

**`src/modules/encoder.py`** - 图编码器 (82行)
- **类**: 
  - `Embedding` - 第12行，特征嵌入层
    - `__init__(sizes, dim_embeddings)` - 第14行，初始化嵌入层
    - `forward(x)` - 第22行，前向传播
  - `Encoder` - 第32行，图神经网络编码器
    - `__init__(dim_in, num_edge_type, dim_hidden, dim_out, t=4)` - 第34行，初始化编码器
    - `embed_node(x, edge_index, edge_attr)` - 第44行，节点嵌入
    - `embed_graph(all_x, graph_ids, node_mask=None)` - 第54行，图级嵌入
    - `forward(batch, return_x=False)` - 第62行，前向传播
- **技术**: GINEConv图神经网络，多层图卷积
- **功能**: 分子图特征编码和图级表示学习

**`src/modules/vae_piece_decoder.py`** - VAE片段解码器 (508行)
- **类**: `VAEPieceDecoder` - 第14行
  - `__init__(atom_embedding_dim, piece_embedding_dim, max_pos, pos_embedding_dim, piece_hidden_dim, node_hidden_dim, num_edge_type, cond_dim, latent_dim, tokenizer, t=4)` - 第16行，初始化解码器
  - `rsample(conds)` - 第50行，重参数化采样
  - `embed_atom(atom_ids, piece_ids, pos_ids)` - 第58行，原子嵌入
  - `forward(x, x_pieces, x_pos, edge_index, edge_attr, pieces, conds, edge_select, golden_edge, return_accu=False)` - 第63行，前向传播
  - `inference(z, max_atom_num, add_edge_th, temperature)` - 第130行，推理生成
- **功能**: 从潜在空间解码生成分子片段
- **核心技术**: 变分自动编码器架构，GRU序列生成
- **特点**: 支持片段级分子生成，边预测

**`src/modules/predictor.py`** - 属性预测器 (22行)
- **类**: `Predictor` - 第5行
  - `__init__(dim_feature, dim_hidden, num_property)` - 第8行，初始化预测器
  - `forward(x)` - 第18行，前向传播
- **功能**: 预测分子性质（QED、LogP、SA等）
- **架构**: 2层MLP + ReLU激活
- **用途**: 联合训练和属性优化

**`src/modules/common_nn.py`** - 通用神经网络组件 (23行)
- **类**: `MLP` - 第7行
  - `__init__(dim_in, dim_hidden, dim_out, act_func, num_layers)` - 第8行，初始化多层感知机
  - `forward(x)` - 第21行，前向传播
- **功能**: 可配置层数的多层感知机
- **复用性**: 被编码器、解码器等模块广泛使用

#### 1.3 PyTorch Lightning模型 (`src/pl_models/`)

**`src/pl_models/ps_vae_model.py`** - PS-VAE主模型 (196行)
- **类**: `PSVAEModel` - 第16行，继承自pl.LightningModule
  - `__init__(config, tokenizer)` - 第17行，初始化主模型
  - `forward(batch, return_accu=False)` - 第28行，前向传播
  - `cal_beta()` - 第50行，计算KL散度权重β
  - `weighted_loss(pred_loss, rec_loss, kl_loss)` - 第58行，加权损失计算
  - `training_step(batch, batch_idx)` - 第62行，训练步骤
  - `validation_step(batch, batch_idx)` - 第75行，验证步骤
  - `validation_epoch_end(outputs)` - 第84行，验证轮次结束
  - `test_step(batch, batch_idx)` - 第102行，测试步骤
  - `configure_optimizers()` - 第107行，优化器配置
  - `sample_z(n, device=None)` - 第114行，采样潜在变量
  - `inference_single_z(z, max_atom_num, add_edge_th, temperature)` - 第119行，单个z推理
  - `inference_batch_z(zs, max_atom_num, add_edge_th, temperature, constraint_mol=None)` - 第123行，批量z推理
  - `inference_single_z_constraint(z, max_atom_num, add_edge_th, temperature, constraint_mol)` - 第127行，约束推理
  - `predict_props(z)` - 第130行，属性预测
- **功能**: 整合编码器、解码器、预测器
- **框架**: PyTorch Lightning
- **特点**: 支持分布式训练、自动混合精度、KL退火

#### 1.4 数据处理模块 (`src/data/`)

**`src/data/bpe_dataset.py`** - BPE数据集处理 (223行)
- **类**: `BPEMolDataset` - 第16行，继承自Dataset
  - `__init__(fname, tokenizer)` - 第17行，初始化数据集
  - `process_step1(mol, tokenizer)` - 第26行，静态方法，分子到原始表示
  - `process_step2(data, tokenizer)` - 第45行，静态方法，到邻接矩阵表示
  - `process_step3(data_list, tokenizer, device='cpu')` - 第75行，静态方法，批处理整理
- **功能**: 分子数据的批处理和预处理
- **特点**: 支持动态批处理、内存优化、缓存机制
- **数据流**: SMILES → 分子对象 → 图表示 → 批处理张量

**`src/data/mol_bpe.py`** - 分子BPE分词器
- **类**: `Tokenizer`
- **功能**: 主子图提取和分子分词
- **算法**: Byte Pair Encoding适配分子结构
- **输出**: 主子图序列和原子分组

**`src/data/split.py`** - 数据集划分
- **功能**: 训练/验证/测试集划分
- **策略**: 随机划分、分层采样
- **支持格式**: SMILES文件

#### 1.5 评估模块 (`src/evaluation/`)

**`src/evaluation/utils.py`** - 评估工具函数 (188行)
- **核心函数**:
  - `smiles2molecule(smiles)` - 第15行，SMILES到分子对象转换
  - `similarity(mol1, mol2)` - 第20行，计算分子间Tanimoto相似度
  - `num_long_cycles(mol)` - 第26行，计算长环数量
  - `get_sa(molecule)` - 第40行，计算合成可达性分数
  - `get_penalized_logp(mol)` - 第46行，计算惩罚LogP值
  - `get_qed(molecule)` - 第82行，计算QED药物相似性
  - `eval_funcs_dict()` - 第87行，评估函数字典
  - `get_normalized_property_scores(mol)` - 第96行，获取归一化属性分数
  - `restore_property_scores(normed_props)` - 第102行，恢复原始属性分数
  - `map_prop_to_idx(props)` - 第111行，属性名到索引映射
  - `overpass_th(prop_vals, prop_idx)` - 第124行，检查是否超过阈值
- **功能**: 
  - 分子性质计算（QED、SA、LogP）
  - 相似性评估（Morgan指纹 + Tanimoto）
  - 有效性检查
  - Guacamol基准测试支持
- **常量**: 
  - `PROP_TH = [0.6, 6.0, 0]` - 第107行，属性阈值
  - `PROPS = ['qed', 'sa', 'logp']` - 第109行，支持的属性列表

#### 1.6 工具模块 (`src/utils/`)

**`src/utils/chem_utils.py`** - 化学计算工具 (506行)
- **类**: 
  - `GeneralVocab` - 第9行，通用化学词汇表
    - `__init__(atom_special=None, bond_special=None)` - 第10行，初始化词汇表
    - `idx_to_atom(idx)` / `atom_to_idx(atom)` - 第22-26行，原子索引转换
    - `idx_to_bond(idx)` / `bond_to_idx(bond)` - 第28-32行，键索引转换
    - `bond_idx_to_valence(idx)` - 第34行，键索引到价态转换
    - `num_atom_type()` / `num_bond_type()` - 第44-48行，类型数量
- **核心函数**:
  - `smiles2molecule(smiles, kekulize=True)` - 第51行，SMILES到分子转换
  - `molecule2smiles(mol)` - 第58行，分子到SMILES转换
  - `rec(mol)` - 第61行，分子重构
  - `data2molecule(vocab, data, sanitize=True)` - 第64行，PyG数据到分子转换
- **常量**: 
  - `MAX_VALENCE` - 第8行，最大价态字典
  - `Bond_List` - 第9行，键类型列表
- **功能**: 
  - SMILES字符串处理
  - 分子对象操作
  - 化学有效性检查
  - 图数据结构转换

**`src/utils/nn_utils.py`** - 神经网络工具
- **类**: 
  - `VAEEarlyStopping` - VAE专用早停机制
- **函数**:
  - `common_config()` - 通用配置
  - `predictor_config()` - 预测器配置
  - `encoder_config()` - 编码器配置
  - `ps_vae_config()` - PS-VAE配置
  - `to_one_hot()` - 独热编码转换
- **功能**: 
  - 配置管理
  - 早停机制
  - 学习率调度
  - 模型检查点

**`src/utils/logger.py`** - 日志工具
- **函数**:
  - `print_log(message, level='INFO')` - 统一日志输出
- **功能**: 统一日志格式和输出，支持不同日志级别

### 2. 独立主子图提取模块 (`ps/`)

**`ps/mol_bpe.py`** - 主子图提取算法
- **类**: `Tokenizer`
  - `__init__(vocab_file=None)` - 初始化分词器
  - `__call__(mol, return_idx=False)` - 分子分词主函数
  - `build_vocab(data_file, vocab_size, output_file)` - 构建词汇表
  - `load_vocab(vocab_file)` - 加载词汇表
  - `save_vocab(vocab_file)` - 保存词汇表
  - `encode(pieces)` - 编码片段序列
  - `decode(indices)` - 解码索引序列
  - `num_piece_type()` - 片段类型数量
  - `pad_idx()` / `end_idx()` - 特殊标记索引
- **功能**: 构建主子图词汇表
- **算法**: 频繁子图挖掘 + Byte Pair Encoding
- **输出**: 主子图词汇文件(.txt格式)

**`ps/molecule.py`** - 分子对象定义
- **类**: 
  - `Molecule` - 分子图表示
    - `__init__(atoms, bonds)` - 初始化分子
    - `add_atom(atom)` - 添加原子
    - `add_bond(bond)` - 添加化学键
    - `to_smiles()` - 转换为SMILES
    - `from_smiles(smiles)` - 从SMILES创建
    - `get_subgraph(atom_indices)` - 获取子图
- **功能**: 分子的子图级表示
- **数据结构**: 节点-边图结构，支持子图操作

**`ps/utils/`** - 工具函数目录
- **`ps/utils/chem_utils.py`** - 化学工具函数
  - `is_valid_mol(mol)` - 分子有效性检查
  - `get_atom_features(atom)` - 原子特征提取
  - `get_bond_features(bond)` - 键特征提取
- **`ps/utils/logger.py`** - 日志工具
  - `setup_logger(name, level)` - 设置日志器
  - `log_info(message)` - 信息日志
  - `log_error(message)` - 错误日志

### 3. 实验脚本 (`scripts/`)

**训练相关脚本**:
- **`scripts/train.sh`** - 模型训练脚本
  - 调用: `src/train.py`
  - 参数: 数据集路径、词汇表、超参数配置
  - 输出: 训练好的模型检查点
- **`scripts/ps_extract.sh`** - 主子图提取脚本
  - 调用: `ps/mol_bpe.py`
  - 参数: 输入数据、词汇表大小、输出路径
  - 输出: 主子图词汇表文件

**属性优化脚本**:
- **`scripts/qed_opt.sh`** - QED属性优化
  - 调用: `src/generate.py`
  - 目标: 优化药物相似性(QED > 0.6)
  - 方法: 梯度上升优化
- **`scripts/plogp_opt.sh`** - PlogP属性优化
  - 调用: `src/generate.py`
  - 目标: 优化惩罚LogP值
  - 约束: 相似性约束
- **`scripts/constraint_prop_opt.sh`** - 约束属性优化
  - 调用: `src/generate.py`
  - 功能: 多属性联合优化
  - 约束: 结构相似性 + 属性阈值

**基准测试脚本**:
- **`scripts/guaca_dist.sh`** - Guacamol分布学习
  - 调用: `src/guacamol_exps/distribution_learning.py`
  - 评估: KL散度、Wasserstein距离
  - 基准: 与真实分子分布的匹配度
- **`scripts/guaca_goal.sh`** - Guacamol目标导向
  - 调用: `src/guacamol_exps/goal_directed/train.py`
  - 评估: 目标分子生成成功率
  - 任务: 特定属性分子设计

### 4. 数据与模型 (`data/`, `ckpts/`)

#### 4.1 数据集 (`data/`)

**ZINC250K数据集**: `zinc250k/`
- **`train.txt`** - 训练集 (200,000分子)
- **`valid.txt`** - 验证集 (25,000分子)
- **`test.txt`** - 测试集 (25,000分子)
- **`processed_data.pkl`** - 预处理数据缓存

**QM9数据集**: `qm9/`
- **特点**: 量子化学性质数据
- **规模**: 134,000小分子
- **性质**: 19种量子化学性质

**词汇表文件**:
- **`zinc_bpe_300.txt`** - ZINC250K主子图词汇(300个)
- **`zinc_bpe_500.txt`** - 扩展词汇表(500个)

#### 4.2 预训练模型 (`ckpts/`)

**ZINC250K模型**:
- **`prop_opt/`** - 属性优化模型
- **`constraint_prop_opt/`** - 约束优化模型
- **`zinc_guaca_dist/`** - 分布学习模型
- **`zinc_guaca_goal/`** - 目标导向模型

**QM9模型**:
- **`qm9_guaca_dist/`** - QM9分布学习
- **`qm9_guaca_goal/`** - QM9目标导向

### 5. 实验分析 (`src/analysis/`, `src/guacamol_exps/`)

#### 5.1 性质相关性分析 (`src/analysis/prop_corr/`)

**`src/analysis/prop_corr/prop_correlation.py`** - 性质相关性分析
- **核心函数**:
  - `analyze_property_correlation(real_mols, gen_mols)` - 计算属性相关性
  - `plot_correlation_heatmap(corr_matrix)` - 绘制相关性热图
  - `compare_distributions(real_props, gen_props)` - 分布对比
  - `statistical_tests(real_data, gen_data)` - 统计显著性检验
- **功能**: 分析生成分子与真实分子的性质分布
- **可视化**: 相关性热图、分布对比图、QQ图
- **输出**: 相关性系数、p值、分布差异报告

#### 5.2 Guacamol基准测试 (`src/guacamol_exps/`)

**`src/guacamol_exps/distribution_learning.py`** - 分布学习实验
- **核心函数**:
  - `evaluate_distribution_learning(model, test_data)` - 分布学习评估
  - `calculate_kl_divergence(real_dist, gen_dist)` - KL散度计算
  - `calculate_wasserstein_distance(real_samples, gen_samples)` - Wasserstein距离
  - `frechet_chemnet_distance(real_mols, gen_mols)` - FCD距离
- **评估指标**: KL散度、Wasserstein距离、FCD
- **基准**: Guacamol分布学习任务

**`src/guacamol_exps/goal_directed/`** - 目标导向实验目录
- **`goal_directed_grad.py`** - 梯度优化方法
  - `gradient_ascent_optimization(model, target_props, n_steps)` - 梯度上升优化
  - `constrained_optimization(model, targets, constraints)` - 约束优化
- **`predictor.py`** - 预测器训练
  - `train_property_predictor(data, props)` - 属性预测器训练
  - `evaluate_predictor(model, test_data)` - 预测器评估
- **`train.py`** - 目标导向训练流程
  - `goal_directed_training(model, objectives)` - 目标导向训练
  - `multi_objective_optimization(model, targets)` - 多目标优化

#### 5.3 统计分析 (`src/statistics/`)

**`src/statistics/enc_space.py`** - 编码空间分析
- **核心函数**:
  - `analyze_latent_space(model, data)` - 潜在空间分析
  - `visualize_latent_space(embeddings, labels)` - 潜在空间可视化
  - `tsne_visualization(embeddings, perplexity=30)` - t-SNE降维可视化
  - `pca_analysis(embeddings, n_components=2)` - PCA主成分分析
  - `interpolation_analysis(model, z1, z2, n_steps)` - 插值分析
- **功能**: 潜在空间可视化和分析
- **方法**: t-SNE、PCA降维、插值分析
- **输出**: 降维可视化图、聚类分析、插值轨迹

**`src/statistics/prop_stats.py`** - 性质统计
- **核心函数**:
  - `calculate_property_statistics(molecules)` - 计算属性统计
  - `plot_property_distributions(prop_data)` - 绘制属性分布
  - `generate_statistics_report(stats_dict)` - 生成统计报告
  - `compare_datasets(dataset1, dataset2)` - 数据集对比
- **功能**: 分子性质分布统计
- **输出**: 统计报告、分布图、箱线图、直方图

## 技术依赖索引

### 核心依赖
```
pytorch==1.8.1+cu101          # 深度学习框架
pytorch-lightning==1.5.7      # 训练框架
torch-geometric==2.1.0.post1   # 图神经网络
rdkit-pypi==2022.3.5          # 化学计算
```

### 图神经网络组件
```
torch-cluster==1.6.0          # 图聚类
torch-scatter==2.0.9          # 散射操作
torch-sparse==0.6.12          # 稀疏矩阵
torch-spline-conv==1.2.1      # 样条卷积
```

### 其他工具
```
networkx==2.8.6               # 图算法
joblib==1.2.0                 # 并行计算
tqdm==4.64.1                  # 进度条
```

## 使用流程索引

### 步骤1: 环境配置
```bash
# 创建conda环境
conda create -n ps-vae python=3.8
conda activate ps-vae

# 安装依赖
pip install -r src/requirements.txt
```

### 步骤2: 主子图提取
```bash
# 从分子数据集提取主子图词汇
python data/mol_bpe.py \
    --data ../data/zinc250k/train.txt \
    --output ../data/zinc_bpe_300.txt \
    --vocab_size 300
```

### 步骤3: 模型训练
```bash
# 训练PS-VAE模型
python train.py \
    --train_set ../data/zinc250k/train.txt \
    --valid_set ../data/zinc250k/valid.txt \
    --test_set ../data/zinc250k/test.txt \
    --vocab ../data/zinc_bpe_300.txt \
    --props qed logp \
    --epochs 6
```

### 步骤4: 分子生成
```bash
# 生成具有优化性质的分子
python generate.py --eval \
    --ckpt ../ckpts/zinc250k/prop_opt/epoch5.ckpt \
    --props qed \
    --n_samples 10000 \
    --output_path qed_optimized.smi
```

### 步骤5: 评估分析
```bash
# 运行Guacamol基准测试
python guacamol_exps/distribution_learning.py
python guacamol_exps/goal_directed/train.py
```

## 配置文件索引

### 环境配置
- **`配置PSVAE指令.txt`** - 详细环境配置说明
- **`src/requirements.txt`** - Python依赖列表
- **`ps/requirements.txt`** - 主子图提取依赖

### 模型配置
- **训练配置**: `src/utils/nn_utils.py`中的配置函数
- **网络配置**: 各模块的超参数设置
- **数据配置**: 数据集路径和预处理参数

## 开发指南

### 代码结构原则
1. **模块化设计**: 每个功能独立成模块
2. **接口统一**: 使用统一的数据接口
3. **配置分离**: 超参数与代码分离
4. **文档完整**: 每个函数都有详细注释

### 扩展指南
1. **添加新属性**: 在`evaluation/utils.py`中添加计算函数
2. **新优化算法**: 在`generate.py`中添加优化方法
3. **新数据集**: 在`data/`目录下添加数据处理脚本
4. **新网络结构**: 在`modules/`目录下添加新模块

### 调试指南
1. **日志系统**: 使用`utils/logger.py`统一日志
2. **可视化**: 使用`statistics/`模块进行分析
3. **单元测试**: 为关键函数编写测试
4. **性能分析**: 使用PyTorch Profiler分析性能

## 时间线

### 第一阶段 (1-2周): 环境搭建
- [ ] 配置开发环境
- [ ] 安装所有依赖
- [ ] 验证数据集完整性
- [ ] 运行基础测试

### 第二阶段 (2-3周): 模型理解
- [ ] 阅读核心代码
- [ ] 理解模型架构
- [ ] 运行预训练模型
- [ ] 分析实验结果

### 第三阶段 (3-4周): 实验复现
- [ ] 复现论文实验
- [ ] 验证基准测试结果
- [ ] 分析性能指标
- [ ] 优化训练流程

### 第四阶段 (4-6周): 扩展开发
- [ ] 实现新功能
- [ ] 优化算法性能
- [ ] 添加新评估指标
- [ ] 完善文档

## 资源需求

### 硬件要求
- **GPU**: NVIDIA RTX 3090或更高 (建议24GB显存)
- **内存**: 32GB RAM (处理大型数据集)
- **存储**: 100GB可用空间 (数据集和模型)

### 软件要求
- **操作系统**: Linux/Windows 10+
- **Python**: 3.8+
- **CUDA**: 11.0+ (GPU加速)
- **Docker**: 可选，用于环境隔离

### 数据资源
- **ZINC250K**: 250,000分子SMILES
- **QM9**: 134,000量子化学数据
- **预训练模型**: 约2GB模型文件

## 联系信息

- **项目主页**: [GitHub Repository]
- **论文链接**: [NeurIPS 2022 Paper]
- **技术支持**: jackie_kxz@outlook.com
- **文档更新**: 定期更新，版本控制

## API参考手册

### 核心API接口

#### PSVAEModel类 (`src/pl_models/ps_vae_model.py`)

```python
class PSVAEModel(pl.LightningModule):
    def __init__(config, tokenizer):
        """初始化PS-VAE模型
        Args:
            config: 模型配置字典
            tokenizer: BPE分词器实例
        """
    
    def sample_z(n, device=None):
        """采样潜在变量
        Args:
            n: 采样数量
            device: 设备类型
        Returns:
            torch.Tensor: 潜在变量张量 [n, latent_dim]
        """
    
    def inference_single_z(z, max_atom_num=38, add_edge_th=0.5, temperature=1.0):
        """单个潜在变量推理
        Args:
            z: 潜在变量 [latent_dim]
            max_atom_num: 最大原子数
            add_edge_th: 边添加阈值
            temperature: 采样温度
        Returns:
            str: 生成的SMILES字符串
        """
```

#### Tokenizer类 (`ps/mol_bpe.py`)

```python
class Tokenizer:
    def __call__(mol, return_idx=False):
        """分子分词
        Args:
            mol: RDKit分子对象
            return_idx: 是否返回索引
        Returns:
            list: 主子图片段列表
        """
    
    def build_vocab(data_file, vocab_size, output_file):
        """构建词汇表
        Args:
            data_file: 输入SMILES文件路径
            vocab_size: 词汇表大小
            output_file: 输出词汇表路径
        """
```

#### 评估函数 (`src/evaluation/utils.py`)

```python
def get_qed(molecule):
    """计算QED药物相似性
    Args:
        molecule: RDKit分子对象
    Returns:
        float: QED分数 [0, 1]
    """

def get_penalized_logp(mol):
    """计算惩罚LogP值
    Args:
        mol: RDKit分子对象
    Returns:
        float: 惩罚LogP分数
    """

def similarity(mol1, mol2):
    """计算分子相似性
    Args:
        mol1, mol2: RDKit分子对象
    Returns:
        float: Tanimoto相似性 [0, 1]
    """
```

### 命令行接口

#### 训练命令
```bash
python src/train.py \
    --train_set <训练集路径> \
    --valid_set <验证集路径> \
    --test_set <测试集路径> \
    --vocab <词汇表路径> \
    --props <属性列表> \
    --epochs <训练轮数> \
    --batch_size <批大小> \
    --lr <学习率> \
    --gpus <GPU数量>
```

#### 生成命令
```bash
python src/generate.py \
    --ckpt <模型检查点路径> \
    --props <目标属性> \
    --n_samples <生成数量> \
    --output_path <输出路径> \
    --max_atom_num <最大原子数> \
    --temperature <采样温度>
```

## 数据格式详解

### 输入数据格式

#### SMILES文件格式
```
# 标准SMILES格式 (train.txt, valid.txt, test.txt)
CCO
CCN
C1CCCCC1
...
```

#### 带属性的SMILES格式
```
# 包含分子属性 (用于属性预测训练)
CCO,0.67,2.31,1.2
CCN,0.45,1.87,0.9
# 格式: SMILES,QED,LogP,SA
```

#### BPE词汇表格式
```
# zinc_bpe_300.txt
<pad>
<end>
C
N
O
[C@@H]
[C@H]
c1ccccc1
...
```

### 输出数据格式

#### 生成分子输出
```
# 生成的SMILES文件
CCOC(=O)c1ccc(N)cc1
CN1CCN(c2ccc(Cl)cc2)CC1
...
```

#### 评估结果格式
```json
{
    "validity": 0.95,
    "uniqueness": 0.87,
    "novelty": 0.73,
    "fcd_score": 2.34,
    "property_stats": {
        "qed_mean": 0.62,
        "logp_mean": 2.45,
        "sa_mean": 3.21
    }
}
```

## 模型架构详解

### 整体架构流程
```
输入分子 → BPE分词 → 图编码器 → 潜在空间 → VAE解码器 → 输出分子
    ↓           ↓         ↓         ↓         ↓
  SMILES    主子图    图嵌入    z向量    重构分子
    ↓
属性预测器 → 分子属性
```

### 关键组件详解

#### 1. 图编码器 (Encoder)
- **输入**: 分子图 (节点特征 + 边特征)
- **网络**: 4层GINEConv图卷积
- **输出**: 图级嵌入向量 [batch_size, hidden_dim]

#### 2. VAE解码器 (VAEPieceDecoder)
- **潜在空间**: 高斯分布 N(μ, σ²)
- **解码过程**: GRU序列生成 + 边预测
- **输出**: 重构的分子图

#### 3. 属性预测器 (Predictor)
- **输入**: 图级嵌入向量
- **网络**: 2层MLP
- **输出**: 分子属性预测值

### 损失函数组成
```python
total_loss = α * reconstruction_loss + β * kl_loss + γ * property_loss
```
- **重构损失**: 原子类型 + 边预测交叉熵
- **KL散度**: 潜在分布与先验分布的差异
- **属性损失**: 预测属性与真实属性的MSE

## 性能优化指南

### 训练优化

#### 1. 内存优化
```python
# 梯度累积减少显存占用
accumulate_grad_batches = 4

# 混合精度训练
precision = 16

# 动态批大小
batch_size = "auto"  # 自动调整批大小
```

#### 2. 速度优化
```python
# 多GPU训练
strategy = "ddp"  # 分布式数据并行
gpus = [0, 1, 2, 3]  # 使用多个GPU

# 数据加载优化
num_workers = 8  # 增加数据加载进程
pin_memory = True  # 固定内存
```

#### 3. 超参数调优
```python
# 学习率调度
scheduler = {
    "scheduler": "ReduceLROnPlateau",
    "monitor": "val_loss",
    "factor": 0.5,
    "patience": 5
}

# KL退火策略
beta_schedule = "linear"  # linear, cosine, constant
beta_warmup_steps = 1000
```

### 推理优化

#### 1. 批量生成
```python
# 批量推理提高效率
batch_size = 64
zs = model.sample_z(batch_size)
mols = model.inference_batch_z(zs)
```

#### 2. 缓存机制
```python
# 缓存词汇表和模型
tokenizer = Tokenizer(vocab_file)  # 只加载一次
model = load_model(ckpt_path)  # 模型复用
```

## 常见问题解答 (FAQ)

### Q1: 训练时显存不足怎么办？
**A**: 
1. 减小批大小: `batch_size = 16`
2. 启用梯度累积: `accumulate_grad_batches = 4`
3. 使用混合精度: `precision = 16`
4. 减少最大原子数: `max_atom_num = 30`

### Q2: 生成的分子无效率高怎么办？
**A**: 
1. 调整采样温度: `temperature = 0.8`
2. 修改边阈值: `add_edge_th = 0.6`
3. 增加训练数据质量
4. 调整KL权重β值

### Q3: 如何添加新的分子属性？
**A**: 
1. 在`src/evaluation/utils.py`中添加计算函数
2. 更新`PROPS`列表
3. 修改预测器输出维度
4. 重新训练模型

### Q4: 模型收敛慢怎么办？
**A**: 
1. 调整学习率: `lr = 1e-3`
2. 使用学习率调度器
3. 增加预热步数: `warmup_steps = 1000`
4. 检查数据质量和分布

### Q5: 如何处理大型数据集？
**A**: 
1. 使用数据流式加载
2. 启用数据并行: `strategy = "ddp"`
3. 增加数据加载进程: `num_workers = 16`
4. 使用SSD存储加速I/O

## 错误处理机制

### 常见错误类型

#### 1. 数据相关错误
```python
# 无效SMILES处理
try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
except Exception as e:
    logger.warning(f"Skipping invalid molecule: {e}")
    continue
```

#### 2. 模型相关错误
```python
# 检查点加载错误
try:
    model = PSVAEModel.load_from_checkpoint(ckpt_path)
except Exception as e:
    logger.error(f"Failed to load checkpoint: {e}")
    raise RuntimeError("Model loading failed")
```

#### 3. 生成相关错误
```python
# 生成失败处理
max_retries = 3
for attempt in range(max_retries):
    try:
        mol = model.inference_single_z(z)
        if mol is not None:
            break
    except Exception as e:
        logger.warning(f"Generation attempt {attempt+1} failed: {e}")
else:
    logger.error("All generation attempts failed")
```

### 日志配置
```python
# 配置详细日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ps_vae.log'),
        logging.StreamHandler()
    ]
)
```

## 实际使用案例

### 案例1: 药物分子优化
```python
# 目标: 优化QED和LogP
target_props = {'qed': 0.8, 'logp': 2.5}
constraint_mol = 'CCO'  # 约束分子

# 加载模型
model = load_model('ckpts/zinc250k/prop_opt/epoch5.ckpt')

# 优化生成
optimized_mols = []
for _ in range(1000):
    z = model.sample_z(1)
    mol = direct_optimize(
        model, z, target_props, 
        lr=0.01, max_iter=100,
        constraint_mol=constraint_mol
    )
    if mol:
        optimized_mols.append(mol)

print(f"Generated {len(optimized_mols)} optimized molecules")
```

### 案例2: 分子库扩展
```python
# 基于种子分子生成相似分子库
seed_smiles = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'  # 布洛芬
seed_mol = Chem.MolFromSmiles(seed_smiles)

# 编码种子分子
z_seed = model.encode_molecule(seed_mol)

# 在潜在空间中采样邻近点
similar_mols = []
for _ in range(100):
    # 添加小幅噪声
    z_noise = z_seed + torch.randn_like(z_seed) * 0.1
    mol = model.inference_single_z(z_noise)
    
    if mol and similarity(seed_mol, mol) > 0.6:
        similar_mols.append(mol)

print(f"Generated {len(similar_mols)} similar molecules")
```

### 案例3: 批量属性预测
```python
# 批量预测分子属性
smiles_list = ['CCO', 'CCN', 'CCC', ...]  # 待预测分子

results = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 计算各种属性
        qed = get_qed(mol)
        logp = get_penalized_logp(mol)
        sa = get_sa(mol)
        
        results.append({
            'smiles': smiles,
            'qed': qed,
            'logp': logp,
            'sa': sa
        })

# 保存结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('molecule_properties.csv', index=False)
```

## 版本兼容性说明

### Python版本支持
- **推荐**: Python 3.8.x
- **支持**: Python 3.7+ (部分功能可能受限)
- **不支持**: Python 3.6及以下

### PyTorch版本兼容性
```
# 推荐组合
PyTorch 1.8.1 + CUDA 11.1
PyTorch 1.9.0 + CUDA 11.1
PyTorch 1.10.0 + CUDA 11.3

# 最低要求
PyTorch >= 1.7.0
```

### RDKit版本说明
```
# 推荐版本
rdkit-pypi==2022.3.5

# 兼容版本
rdkit-pypi>=2021.9.0

# 注意: 不同版本的RDKit可能影响分子属性计算结果
```

### 操作系统兼容性
- **Linux**: 完全支持 (推荐Ubuntu 18.04+)
- **Windows**: 支持 (Windows 10+)
- **macOS**: 部分支持 (可能需要额外配置)

## 扩展开发指南

### 添加新的神经网络层
```python
# 在 src/modules/ 下创建新模块
class CustomLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 在主模型中集成
class PSVAEModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        # ... 其他初始化代码
        self.custom_layer = CustomLayer(config.hidden_dim, config.output_dim)
```

### 添加新的损失函数
```python
# 在 src/utils/ 下添加损失函数
def custom_loss(pred, target, weight=1.0):
    """自定义损失函数"""
    loss = F.mse_loss(pred, target)
    return weight * loss

# 在模型中使用
def training_step(self, batch, batch_idx):
    # ... 现有代码
    custom_loss_val = custom_loss(pred, target)
    total_loss += custom_loss_val
```

### 添加新的评估指标
```python
# 在 src/evaluation/utils.py 中添加
def custom_metric(mol):
    """自定义分子评估指标"""
    # 实现具体计算逻辑
    return score

# 更新评估函数字典
eval_funcs_dict = {
    'qed': get_qed,
    'logp': get_penalized_logp,
    'sa': get_sa,
    'custom': custom_metric  # 新增指标
}
```

---

*本索引方案提供了PS-VAE项目的完整技术地图，涵盖了从数据处理到模型训练，从分子生成到性质优化的全流程代码组织结构，以及详细的API参考、使用案例和扩展指南。*