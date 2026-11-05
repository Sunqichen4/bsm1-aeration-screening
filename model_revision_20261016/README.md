# M3：连续氨氮标定、参数重构与冻结后独立验证

本目录是2026年10月16日建立的新研究版本，用于检查：减少冗余参数后，连续峰值标定能否改善氨氮浓度预测与合规筛选，以及耗氧反馈在匹配的消融比较中是否仍有效。旧版M1/M2及其结果保留原身份；M3不把旧244点或96点重新作为独立测试。

本次已完成114项测试、90个新原生BSM1参考及6个半步长复算，并通过原始轨迹审计。主要结果与解释见`M3_修改与验证报告.md`及`independent_results_review.md`。`design.json`说明研究问题和解释范围，`protocol.json`固定拟合、选点、Gate与验证规则。

## 比较对象与解释范围

| 模型 | 标定/作用 | 可以支持的比较 |
|---|---|---|
| `full_calibrated` | 连续峰值标定，6个自由参数，包含可标定耗氧比ρ | 检查连续任务标定及额外灵活性带来的整体变化 |
| `full_fixed_ratio` | 连续峰值标定，固定ρ，5个自由参数 | 与`no_feedback`构成匹配的5参数反馈消融对照 |
| `no_feedback` | 连续峰值标定，去除对应耗氧反馈，5个自由参数 | 与`full_fixed_ratio`比较反馈作用 |
| `poly_ridge` | 连续峰值，多项式Ridge；三开发工况留一交叉验证 | 同连续目标监督下的统计回归对照 |
| `rbf_kernel_ridge` | 连续峰值，RBF核Ridge；三开发工况留一交叉验证 | 同连续目标监督下的非线性统计回归对照 |
| `legacy_M2_binary` | 保留原M2的二元监督参数，用于相同新候选上的历史背景比较 | 比较重构前后整体行为；不属于同连续监督的公平比较 |

五个新模型使用相同750个开发案例的`SNH_max_eval`连续目标。相同目标监督不意味着相同先验、模型容量或计算预算：过程模型保留历史经验生物量闭合，统计回归只使用负荷坐标L和有效传氧`z = α × KLa`；过程模型优化与统计模型交叉验证的预算也不同。6参数完整模型与5参数去反馈模型之间的差异，同时包含额外自由度的影响，反馈消融结论应使用上述匹配的5参数对。

ρ剖面损失和Jacobian秩用于检查局部辨识性与补偿关系，不能据此报告生理常数、物理参数置信区间或物理机制已被验证。

## 环境和路径

历史核验使用Windows 64位Python 3.11.15。以下以可调用的Python 3.11解释器`python`为例。脚本使用NumPy、SciPy、Numba等，不需要PyTorch训练。`requirements-lock.txt`列出本工作流所需包及运行环境中查到的版本，包括测试和作图依赖；它不代表原工作环境中的所有包。

以下PowerShell示例从仓库根目录进入本目录，随后运行测试；其他命令均在本目录执行，脚本和数据使用相对路径：

```powershell
Set-Location -LiteralPath '.\model_revision_20261016'
$m3Python = 'python'
& $m3Python -m pytest .\tests -q
```

若另建Python 3.11环境，可用该环境的解释器执行`-m pip install -r .\requirements-lock.txt`。安装版本一致仍须通过测试与结果审计；本地原生依赖和平台差异可能影响数值及计时。

## 核查已经生成的结果

以下命令要求对应运行已经完成。`--audit`读取已保存轨迹、核对来源哈希并独立重算指标，不调用新的BSM1积分。`analyze_results.py`首先验证冻结文件并调用独立审计，任何覆盖缺失、失败状态或哈希不一致均终止分析。

```powershell
& $m3Python .\fresh_reference.py . --audit
& $m3Python .\analyze_results.py .
& $m3Python .\fresh_reference.py . --audit-half-step
& $m3Python .\make_figures.py
```

主参考通过的判据是`reference_audit.json`中的`passed: true`。半步长核查有独立的`numerical/half_step_audit.json`。`analyze_results.py`生成：

- `evaluation_joined.csv`：冻结预测、原始分类、Gate路由和审计后参考逐点对应。
- `metrics_overall.csv`：各模型的浓度误差、原始误放/误拒、核验数、纠正/遗漏错误及最终错误。
- `metrics_by_condition.csv`：逐工况、逐模型的相同指标。
- `boundaries.csv`：采样候选中的首合规KLa及其差值，符号为模型减参考。
- `analysis_summary.json`：来源哈希、审计状态、指标定义及开发诊断背景。

表格分别报告`all`、固定`coarse`和`prediction_enriched`三层。加密候选由五个新模型的预测边界选出；历史模型不参与选点。混合采样结果不表示实际运行工况的自然分布；首合规采样候选也不等于连续最优KLa。

### 路径迁移与逐值重放

`verification/portability_check.py`是冻结后的复现检查器：将冻结代码、输入、模型与原始参考复制到含中文和空格的新路径，再审计主/半步长参考并重放540个模型预测及Gate分支。它不重新拟合、不产生新的原生参考标签。运行时使用新的检查目录名：

```powershell
& $m3Python .\verification\portability_check.py --run-name 'relocation smoke new'
```

逐值复现输入应以`pd.read_csv(..., float_precision="round_trip")`读取，并在导入NumPy前设置单线程。已观察到默认CSV解析使15行α产生约1e-16的末位差异，RBF的大权重将其放大为约1.02e-8 mg N/L预测差；本次分类不受影响。精确保留浮点数后540个预测值及全部分类/Gate分支完全一致。证据见`portability_validation.json`及`verification/replay_diagnostic_*.json`，首次严格检查的失败记录也已保留。

## 在新目录完整重跑

保留当前目录的开发run、日志、`freeze.json`、参考JSON、原始NPZ轨迹及分析结果。重新拟合必须使用一个新的目录和新run；脚本会拒绝覆盖已有正式开发目录或冻结文件。不要在旧结果上更换模型或修改Gate后继续沿用旧冻结身份。

下面复制输入和代码到一个新的版本目录，原目录及其所有运行记录保持可追溯；目标目录应尚不存在：

```powershell
$m3Source = (Get-Location).Path
$m3Replay = Join-Path (Split-Path $m3Source -Parent) 'model_revision_replay_01'
if (Test-Path -LiteralPath $m3Replay) { throw '目标目录已存在，请使用新的版本目录。' }
New-Item -ItemType Directory -Path $m3Replay | Out-Null
Get-ChildItem -LiteralPath $m3Source -File | Where-Object {
    $_.Extension -eq '.py' -or $_.Name -in @('README.md','requirements-lock.txt','design.json','protocol.json','source_manifest.json')
} | Copy-Item -Destination $m3Replay
foreach ($m3Folder in @('data','legacy','tests')) {
    Copy-Item -LiteralPath (Join-Path $m3Source $m3Folder) -Destination $m3Replay -Recurse
}
Set-Location -LiteralPath $m3Replay
```

不要把旧`outputs`、冻结候选、`freeze.json`、`reference*`或`numerical`当作新run的输入。新目录内执行：

```powershell
& $m3Python -m pytest .\tests -q
& $m3Python .\run_study.py fit
& $m3Python .\run_study.py freeze
& $m3Python .\run_study.py references --workers 4
& $m3Python .\fresh_reference.py . --audit
& $m3Python .\fresh_reference.py . --run-half-step --workers 2
& $m3Python .\fresh_reference.py . --audit-half-step
& $m3Python .\analyze_results.py .
& $m3Python .\make_figures.py
```

顺序及产物如下：

1. `fit`只读750开发案例，拟合模型、进行统计模型交叉验证、辨识性诊断与ρ剖面，并保存开发误差及经验缓冲宽度。检查`outputs/development/development_checks.json`的`pass`后继续。
2. `freeze`在获得任何新标签前固定模型、六个新恒定工况、候选集合、预测、Gate路由和半步长核查身份，并写入SHA-256清单。拟合起点时的科学源哈希须与冻结时一致。
3. `references`在六个新工况中，为所有冻结候选启动独立原生BSM1实例。主参考步长1分钟、总时长67天，报告窗口为输出时刻`60 ≤ t < 67`天。原始轨迹与执行来源一并保存。
4. `--run-half-step`仅对预先冻结的每工况一个候选，使用0.5分钟重新积分，输出到`numerical/`。这是数值核查，不替换主参考标签，也不重新选点。
5. 通过审计后生成分析表和图。保持负面结果及失败记录。并行参考的墙钟耗时与工作量用于记录本次计算成本，不构成与旧串行基线的速度比较。

同一冻结run若被中断，可在代码、依赖、冻结及已有轨迹均未变化时再次执行相同参考命令；程序先核查已有检查点，只补尚未完成的候选。失败参考不会被静默改为成功。若要修改科学实现或重新拟合，使用另一个新目录并生成新冻结。

## 本版本Gate与适用范围

M3统一采用连续浓度残差缓冲：每个模型的`b`为750开发案例上最大绝对峰值残差。域内若`prediction + b < 4`则直接放行，若`prediction - b > 4`则直接拒绝，否则参考核验；输入超出预设联合/特征域也进入核验。这是经验规则，未校准名义覆盖概率。所有候选均另有独立科学参考，因此表中的`gate_selected`用于模拟部署核验成本；科学评估本身的参考计算没有被省去。

`legacy_M2_binary`显示的Gate也使用该新缓冲包装，因此不能称作旧M2的KLa Gate复现。保留历史模型参数不等于保留历史Gate协议。

新验证限于15°C、六个恒定工况和固定曝气候选。BSM1是模拟参考，没有真实污水厂测量验证，也没有建立低温、动态扰动、现场能耗或任意工况外推有效性。工程边界、经验闭合和数值诊断应与已验证的范围一起报告。

`legacy/`为本版本所需历史科学实现及第三方来源副本；请保留其中的`THIRD_PARTY_NOTICES.md`及许可证/来源记录。`source_manifest.json`用于记录原工作区来源和受保护文件，不是新参考标签数据集。
