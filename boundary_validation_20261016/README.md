# M3B：冻结模型的阈值边界补充验证

本目录补充验证原M3在真实NH₄-N阈值附近的表现。M3参数、开发残差缓冲、Gate和4 mg N/L判据全部保持固定；本轮不训练模型。六个新恒定工况、108个固定有效传氧网格点、12个legacy预测加密点，以及12个半步长核查点均在新原生参考产生前冻结。

**来源与统计口径**：已见M3的90例用于确定本轮有效传氧范围。108个固定点作为主要模型比较；legacy M2稳态交叉引导的12点单独报告。`all`仅为本轮120点的描述性汇总，不能与旧90点混为一个无偏总体测试集。六工况并非独立同分布现场样本。参考结果无论支持哪个模型均保留，不在结果后扩网或调参。

## 文件入口

- `M3B_边界补充验证报告.md`：实际结果、数值稳定性、可支持的结论及下一步。
- `M3B_论文补充段落.md`：可并入论文的方法、结果、讨论及图注，并附证据和修改记录。
- `protocol.json`、`design.json`：冻结前协议及待验证问题。
- `freeze.json`：输入、参数、预测、Gate路由、原生源码与配置的SHA-256清单。
- `new_candidates.csv`：108个`coarse`主网格点和12个`prediction_enriched`点；`effective_kla`单位d⁻¹。
- `frozen_predictions.csv`：6模型×120点的原始有限积分/回归预测、数值诊断及Gate路由。
- `source_historical_input_coordinates.csv`：只含历史输入，用于查重复，未读取历史标签进行本轮选点。
- `reference/`、`reference_trajectories/`：120次新原生BSM1积分的JSON及完整保存的评价轨迹NPZ。
- `reference_provenance.json`、`reference_execution.json`、`reference_audit.json`：运行前来源、成本与原始独立审计。
- `numerical/`：12次新0.5 min积分、比较和原始独立审计；主标签仍为1 min。
- `metrics_overall.csv`、`metrics_by_condition.csv`：浓度、原始误判、Gate纠正与遗漏、核验成本，按三层分别报告。
- `coverage_by_condition.csv`：各工况真实阈值两侧和阈值距离，主层必须独立阅读。
- `near_threshold_metrics.csv`、`matched_pair_by_condition.csv`：阈值邻域结果及匹配5参数模型对照。
- `figures/`：原始绘图数据、图注和PNG、600 dpi TIFF、SVG、PDF。
- `tests_results.xml`、`tests_run.log`、`verification/`：轻量测试、独立统计复算与保护原文件核对。

## 运行环境与命令

历史核验环境为Windows x86-64、Python 3.11.15。以下以可调用的Python 3.11解释器`python`为例；依赖版本见`requirements-lock.txt`，不需要PyTorch。以下命令均在本目录运行。

```powershell
# 验证已冻结数据和参考原始轨迹（不重新积分）
python -B boundary_study.py audit
python -B fresh_reference.py . --audit-half-step

# 重新计算720个预测并与冻结值逐值比较
python -B boundary_study.py replay

# 从审计通过的参考重新生成统计与科研图
python -B boundary_analysis.py
python -B make_boundary_figures.py

# 轻量设计和完整性测试，不运行原生BSM1长积分
python -B -m pytest tests -q
python -B verification/independent_review.py
```

独立审核器默认也检查相邻的原M3目录及更早原文件保护记录。在外部机器仅有本补充包时，可显式加`--skip-external-protection`，仍复算指标、轨迹峰值及包内来源哈希，并将外部原文件保护标记为未核验。

`boundary_study.py main`与`half`分别调用固定4/2进程原生计算，已有检查点只在完整哈希/轨迹审计通过后复用。两者在完整交付目录中通常只复核既有积分；如需从零重算，请复制到独立目录，保留全部冻结文件和源码，省略`reference*`运行产物、`numerical/`与后分析产物，再依次运行`main`和`half`。不要删改本交付的原始结果。`freeze`是本轮一次性构建阶段，已冻结后拒绝覆盖；其历史输入来源路径仅在重新设计新研究时才需要。

生产和测试的Numba缓存分别为`.runtime_cache/production`与`.runtime_cache/tests`，均不打包。所有新科学CSV读取使用`float_precision="round_trip"`。运行入口在导入NumPy前固定每进程1线程；不要把测试禁用JIT的设置带入生产。

## 解释约定

误放：预测合规、参考超限；误拒：预测超限、参考合规。Gate核验后的标签对被核验点使用参考值，需同时报告核验次数，不能把“最终无错”等同于独立预测无错。科学验证对全部120点实际生成参考；Gate次数表示按固定路由模拟部署所需调用数，与此次科学验证实际计算量不同。

匹配结构比较使用5参数`full_fixed_ratio`与5参数`no_feedback`；6参数`full_calibrated`有额外自由度。历史M2二元监督仅作背景对照；其加密层的取点又依赖自身预测。过程模型保留历史生物量闭合，与纯回归的历史先验不同。既有参数触界、辅助SNO投影与物理辨识限制沿用M3，不因本轮边界验证而解除。

原M3目录及ZIP、已有中文初稿不覆盖。本目录是实验补充包，论文全文仍需依据结果整合，不能将M1、M2、M3和M3B不同版本的最佳数字拼接成单一模型的表现。
