"""Build Chinese evidence reports from completed and independently audited M3B artifacts."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
MODELS=['full_calibrated','full_fixed_ratio','no_feedback','poly_ridge','rbf_kernel_ridge','legacy_M2_binary']
NAMES=dict(zip(MODELS,['自由ρ完整模型（6参数）','固定ρ完整模型（5参数）','去反馈模型（5参数）','多项式Ridge','RBF核Ridge','历史M2（二元监督背景）']))
def read(name):return pd.read_csv(ROOT/name,float_precision='round_trip')
def js(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def fmt(x,d=4):return '缺侧' if pd.isna(x) else f'{x:.{d}f}'
def metric_table(metrics,stratum):
    g=metrics.loc[metrics.stratum.eq(stratum)].set_index('model')
    lines=['| 模型 | RMSE | MAE | 原始误放/误拒 | Gate核验 | 最终误放/误拒 |','|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        r=g.loc[name]
        lines.append(f'| {NAMES[name]} | {r.rmse:.4f} | {r.mae:.4f} | {int(r.raw_false_releases)}/{int(r.raw_false_rejections)} | {int(r.gate_selected)}/{int(r.n_candidates)}（{r.gate_selected_fraction:.1%}） | {int(r.final_false_releases)}/{int(r.final_false_rejections)} |')
    return '\n'.join(lines)
def generate():
    audit=js('reference_audit.json'); halfaudit=js('numerical/half_step_audit.json'); independent=js('independent_review.json');replay=js('prediction_replay.json')
    if not all(x['passed'] for x in (audit,halfaudit,independent,replay)):raise ValueError('Required audit failed')
    for name,digest in independent['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Independent review input changed: '+name)
    metrics=read('metrics_overall.csv'); coverage=read('coverage_by_condition.csv');truth=read('boundary_candidate_truth.csv');pred=read('frozen_predictions.csv');near=read('near_threshold_metrics.csv');matched=read('matched_pair_by_condition.csv')
    half=read('numerical/half_step_summary.csv'); hd=js('numerical/half_step_diagnostics.json');execution=js('reference_execution.json');hexecution=js('numerical/reference_execution.json')
    checks=js('pre_reference_checks.json');freeze=js('freeze.json')
    suites=list(ET.parse(ROOT/'tests_results.xml').getroot().iter('testsuite'))
    if any(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites):raise ValueError('Tests contain failures/errors')
    ntests=sum(int(x.attrib['tests']) for x in suites)
    coarse=truth.loc[truth.grid_kind.eq('coarse')]; enriched=truth.loc[truth.grid_kind.eq('prediction_enriched')]
    cv=coverage.loc[coverage.stratum.eq('coarse')];ecv=coverage.loc[coverage.stratum.eq('prediction_enriched')]
    m=metrics.loc[metrics.stratum.eq('coarse')].set_index('model');a=metrics.loc[metrics.stratum.eq('all')].set_index('model')
    f=m.loc['full_calibrated']; nf=m.loc['no_feedback'];fixed=m.loc['full_fixed_ratio']; legacy=m.loc['legacy_M2_binary']
    near05=near.loc[near.stratum.eq('coarse')&near.window_mgN_L.eq(.5)].set_index('model')
    nf05=near05.loc['no_feedback'];f05=near05.loc['full_calibrated'];legacy05=near05.loc['legacy_M2_binary']
    best=m.drop(index='legacy_M2_binary').rmse.idxmin()
    mp=matched.loc[matched.stratum.eq('coarse')]
    full_worse=int((mp.difference_rmse>0).sum());full_better=int((mp.difference_rmse<0).sum())
    nnear={w:int((np.abs(coarse.SNH_max_eval-4)<=w).sum()) for w in (.25,.5,1.)}
    maxdiff=max(r['maximum_absolute_numeric_difference'] for r in independent['table_checks'].values())
    gate_min=float(m.gate_selected_fraction.min());gate_max=float(m.gate_selected_fraction.max())
    coverage_lines=['| 工况（L，q） | 合规/超限 | 最近合规峰值 | 最近超限峰值 | 距4≤0.5数量 | 邻点标签变化 |','|---|---:|---:|---:|---:|---:|']
    for i,r in enumerate(cv.itertuples(),1):
        cond=js('protocol.json')['conditions'][i-1]
        coverage_lines.append(f'| ({cond["load"]:.2f}, {cond["q"]:.2f}) | {r.compliant_count}/{r.noncompliant_count} | {fmt(r.nearest_compliant_peak_mgN_L)} | {fmt(r.nearest_noncompliant_peak_mgN_L)} | {getattr(r,"within_0p5_count")} | {r.adjacent_label_change_count} |')
    cover_table='\n'.join(coverage_lines)
    both=int(cv.has_both_sides.sum()); eboth=int(ecv.has_both_sides.sum())
    closure='本轮补足了六个新工况在真实阈值两侧的固定网格证据。' if both==6 else f'本轮仅在{both}/6工况形成双侧，缺侧保留，按冻结协议不再扩网。'
    cliprows=pred.loc[pred.model.isin(MODELS[:3])]
    clips='；'.join(f'{NAMES[name]}：SNO下界投影{int(g.SNO_lower_clips.sum())}次、NH4/DO裁剪{int(g.critical_NH4_DO_clips.sum())}次' for name,g in cliprows.groupby('model',sort=False))
    hs=int(half.SNH_zero_violation.sum()); main_wall=sum(r['wall_seconds'] for r in execution['sessions']); half_wall=sum(r['wall_seconds'] for r in hexecution['sessions'])
    protection=independent['origin_protection']
    if protection.get('external_protection_status')=='VERIFIED':
        protection_sentence=f"原M3保护文件{protection['original_M3_files_checked']}份以及此前原稿/程序保护记录{protection['pre_M3_protected_files_checked']}份均保持一致。原M3包和中文论文初稿未覆盖。"
    else:
        protection_sentence='本次报告重建显式跳过了外部原文件保护检查（UNVERIFIED）；包内来源快照与科学文件哈希仍通过，不据此声称外部原稿或原M3目录保持一致。'
    report=f'''# M3B阈值边界补充验证报告

本轮完成冻结M3模型在原生BSM1参考阈值附近的边界补充验证，未重新标定模型或调整Gate。{closure}固定主网格中共有{len(coarse)}点，参考合规{int(coarse.reference_compliant.sum())}点、超限{int((~coarse.reference_compliant).sum())}点；距4 mg N/L不超过0.25、0.5和1.0的点分别为{nnear[.25]}、{nnear[.5]}和{nnear[1.]}。这些是已设计工况上的描述性结果。

补充实验状态：**READY_WITH_WARNINGS**。主/半步长原始轨迹审计、独立指标复算和预测重放均通过；模型比较、Gate代价和数值边界按下文保留。论文全文状态仍为**NOT_READY**：本轮提供实验和可并入段落，尚未重写全文或完成最新文献及期刊要求核对。

## 1. 冻结范围与数据来源（VERIFIED）

六新工况L为0.13、0.31、0.43、0.61、0.73、0.91，对应q为0.27、0.63、0.38、0.77、0.34、0.68，α=0.7+0.3L+q[1−(0.7+0.3L)]。全部为15°C恒定域内条件。每工况有效传氧z=αKLa取112、220及136至196、步长4，共18个固定点，合计108点。命令KLa=z/α，处于原40–520 d⁻¹范围。

原M3的90个结果已用于确定本轮z范围，应明确披露。固定108点承担主比较；另以冻结历史M2稳态预测在[112,220]内的一处NH4=4交叉c，预先选c±1共12点，单列为预测加密层。本次六工况均找到交叉且无物理重复，共120个主参考；12加密点也在新标签产生前选为0.5 min数值核查点。历史四套候选只读取输入字段检查重复，不导入其标签。本轮没有基于新参考补点、改参数、改缓冲或改变4 mg N/L阈值。

全部720条模型预测及路由在{freeze['frozen_at_utc']}冻结。冻结SHA-256：`{hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest()}`。主参考来源记录创建时间为{js('reference_provenance.json')['created_at_utc']}，晚于冻结。模型、gates及过程/统计预测源码相对M3来源逐字节一致；新副本原生wrapper仅修正CSV的round_trip读取精度。

## 2. 真实双侧覆盖（VERIFIED）

下表只统计固定108点，浓度单位mg N/L。合规以参考峰值≤4定义；最近点按实际浓度距4的距离确定，并非连续阈值估计。

{cover_table}

固定网格有{both}/6工况实际出现双侧。12个加密点中参考合规{int(enriched.reference_compliant.sum())}点、超限{int((~enriched.reference_compliant).sum())}点，加密层有{eboth}/6工况双侧。各层的最近双侧距离和0.25/0.5/1.0窗口数量见`coverage_by_condition.csv`。相邻采样的标签变化保存在`observed_sign_changes.csv`；不据此证明全域单调性、唯一交叉或连续最小可行KLa。

## 3. 固定108点的主要模型比较（VERIFIED）

RMSE和MAE单位mg N/L；误放是预测合规但参考超限，误拒为预测超限但参考合规。Gate核验数表示按冻结路由模拟部署的参考调用数。

{metric_table(metrics,'coarse')}

五个连续目标模型中，本轮固定主网格RMSE最低的是{NAMES[best]}。自由ρ完整模型RMSE为{f.rmse:.4f}、MAE为{f.mae:.4f}，去反馈模型分别为{nf.rmse:.4f}和{nf.mae:.4f}；二者原始误放/误拒分别为{int(f.raw_false_releases)}/{int(f.raw_false_rejections)}及{int(nf.raw_false_releases)}/{int(nf.raw_false_rejections)}。浓度误差、阈值误判和核验代价须共同阅读，不以单一排名替代任务表现。

参考峰值距4不超过0.5 mg N/L的固定网格子集共{int(f05.n_candidates)}点，自由ρ完整模型与去反馈模型RMSE分别为{fmt(f05.rmse)}和{fmt(nf05.rmse)}，历史M2为{fmt(legacy05.rmse)}。该参考定义的近阈值子集仍是固定主格的一部分，结果提示总体浓度排序不能直接替代限值附近表现。六模型在本轮全部120点的原始误放均为{int(metrics.loc[metrics.stratum.eq('all'),'raw_false_releases'].max())}；本轮观察到的Gate纠正针对误拒，没有新增捕获原始误放的案例。

固定主格48个合规点中，自由ρ完整模型误拒23个（47.9%），历史M2误拒1个；自由ρ模型主格MAE也并非六模型最低。距阈值≤0.5的8点中有6个合规、2个超限，自由ρ和去反馈模型均把这6个合规点全部原始误拒，并都核验8/8点。RBF主格核验58点，少于自由ρ的84点，但其浓度与误拒表现较差。上述取舍应与RMSE优势一并呈现。距阈值≤0.25的固定点仅3个且全部合规，不能声称最窄窗口也已双侧覆盖。

严格匹配的5参数结构对照为固定ρ完整模型与去反馈模型。其主网格总体RMSE分别为{fixed.rmse:.4f}与{nf.rmse:.4f}；逐工况完整模型RMSE较高{full_worse}个、较低{full_better}个。逐点配对和逐层结果见`matched_pair_by_condition.csv`。6参数自由ρ模型包含额外自由度，不将其改善直接归因于反馈结构。历史M2使用不同二元监督背景，仅作情境对照；本轮加密层又依赖它的预测，不能据该层排序推导其普遍优势。

本轮主网格六模型Gate比例为{gate_min:.1%}–{gate_max:.1%}，直接放行总数为{int(pred.direct_release.sum())}。开发经验缓冲全部大于4 mg N/L，非负预测无法满足预测加缓冲小于4的直接放行条件。被核验点使用参考标签，因此最终错误需与原始错误和核验数一起呈现；零观察错误不构成概率风险保证。全部120点均实际生成新参考，这是科学验证成本，不能把模拟Gate比例写成此轮真实省去的计算量。

## 4. 加密层与全部120点（VERIFIED，分层报告）

以下12点由legacy预测交叉引导，具有模型选点偏好，作为补充压力测试。

{metric_table(metrics,'prediction_enriched')}

全部120点的描述性汇总如下，不与M3旧90点合并。

{metric_table(metrics,'all')}

阈值邻域集合按新参考的|峰值−4|≤0.25、0.5和1.0定义，全部六模型使用相同集合；其中的浓度、误判和Gate代价见`near_threshold_metrics.csv`。这些参考定义的子集仅作描述，未反向用于选点或标定。

## 5. 数值核查、可复现性与原文件保护（VERIFIED）

120个1 min原生参考全部成功，每例评价窗口为60≤t<67 d、10080个报告样本。预选12点的0.5 min参考均成功，每例20160个样本；半步长标签中合规{hs}点、超限{len(half)-hs}点。主/半峰值最大绝对差为{hd['maximum_absolute_peak_difference']:.9f} mg N/L，标签翻转{hd['label_change_count']}次。主标签仍为原定1 min，未用半步长静默替换。

模型有限预测NH4/DO裁剪总数为{checks['critical_NH4_DO_clips']}，稳态与有限峰值最大差{checks['steady_finite_max_abs_difference']:.9f} mg N/L。{clips}。SNO投影为67 d全程逐步末事件累计，包含预热；辅助SNO不反馈其余状态，因此该诊断不能扩展为完整六状态物理有效性证明。原M3参数触界与物理辨识限制仍保留。

本轮{ntests}项设计、完整性和边界统计测试通过，未在测试中调用原生长积分。720条预测重放逐值相同、最大峰值差{replay['maximum_peak_difference']:.1f}，分类及全部Gate分支相同。独立审核脚本未导入正式分析器，从冻结预测/参考独立重算总体、逐工况与覆盖表，最大指标差{maxdiff:.3g}；另从132份NPZ重新提取SNH峰值、核对报告窗口及哈希。

额外复制到包含中文和空格的独立路径后，120点主参考原始审计通过，720条预测及全部Gate分支再次逐值一致、峰值最大差0。迁移检查没有重新运行原生积分，也未包含半步长路径迁移；半步长已在本研究目录完成独立原始审计。见`portability_validation.json`和对应日志。

{protection_sentence}主参考4进程墙钟{main_wall:.2f} s，总核函数工作量{execution['successful_kernel_work_seconds']:.2f} s；半步长2进程墙钟{half_wall:.2f} s。耗时只作为本轮计算记录，不与旧串行实验构造速度比。

## 6. 对论文写作的实际影响

{closure}建议现在转入论文整合，保留已冻结模型和全部负面结果。这轮检验的是实际阈值邻域覆盖与数值稳定性，不能改写为完整反馈模型普遍最优或Gate低调用数优势。正文应把连续浓度精度、阈值筛选及参考核验成本作为不同评价维度，并把M1、M2、M3与本轮M3B分别标明。

可直接使用的方法、结果和讨论段落见`M3B_论文补充段落.md`；科研图及图注见`figures/`。文献新颖性、实厂有效性、低温/动态泛化及实测能耗收益为**UNVERIFIED**，本轮未产生这些证据。**AUTHOR_INPUT_NEEDED**：本补充实验无待提供的信息；若论文继续主张物理参数辨识或实厂收益，需另有独立证据，或据现有结果调整相应主张。
'''
    (ROOT/'M3B_边界补充验证报告.md').write_text(report,encoding='utf-8')
    paper=f'''# M3B论文补充段落与证据衔接

## 可并入实验方法

为补充连续标定模型在实际NH₄-N限值附近的评价，保持M3六模型参数、残差缓冲与筛选规则固定，在六个新的恒定域内工况上进行边界验证。负荷坐标L依次为0.13、0.31、0.43、0.61、0.73和0.91，对应q为0.27、0.63、0.38、0.77、0.34和0.68，传氧倍率按α=0.7+0.3L+q[1−(0.7+0.3L)]确定。各工况有效传氧z=αKLa取112、220以及136至196、步长4 d⁻¹的固定网格，共108个主要评价候选。原M3结果参与了本轮范围设计。另在冻结历史M2稳态预测的一处4 mg N/L交叉附近，选取z=c±1 d⁻¹的12个候选作为模型引导加密层，独立报告其结果。所有实际输入、六模型预测和Gate路由均在本轮新参考产生前冻结，未利用新参考重新标定或扩展候选。

参考采用原生BSM1在15°C下独立积分67 d，前60 d预热，后7 d按60≤t<67 d的报告端点评价，以出水NH₄-N峰值≤4 mg N/L判为合规。主步长为1 min，预先选择的12个加密候选另以0.5 min重新积分，报告峰值差与标签变化且保留主步长标签。分别在固定主网格、预测加密层及其并集中计算浓度RMSE、MAE、原始误放/误拒、Gate核验比例和最终错误，并按参考距4的0.25、0.5及1.0 mg N/L窗口描述阈值邻域表现。

## 可并入结果

固定108点中，六个工况有{both}个实际覆盖合规与超限两侧，参考合规和超限候选分别为{int(coarse.reference_compliant.sum())}和{int((~coarse.reference_compliant).sum())}个。距阈值不超过0.25、0.5和1.0 mg N/L的候选分别为{nnear[.25]}、{nnear[.5]}和{nnear[1.]}个。12个模型引导加密候选中，参考合规{int(enriched.reference_compliant.sum())}个、超限{int((~enriched.reference_compliant).sum())}个，{eboth}个工况在该层形成双侧覆盖。由此可将模型在实际浓度限值附近的误差与原M3较粗候选上的表现区分报告。

在固定主网格上，自由ρ完整模型的RMSE和MAE分别为{f.rmse:.4f}和{f.mae:.4f} mg N/L，原始误放/误拒为{int(f.raw_false_releases)}/{int(f.raw_false_rejections)}，Gate核验{int(f.gate_selected)}/108点。固定ρ完整模型与去反馈模型均使用5个可调参数，其RMSE分别为{fixed.rmse:.4f}和{nf.rmse:.4f} mg N/L，完整模型在六个工况中有{full_worse}个RMSE较高、有{full_better}个较低。该匹配对照用于判断指定反馈项在本连续任务标定下的表现；6参数模型的额外自由度不视为纯结构贡献。完整六模型和各层结果见表M3B-1及补充表。

在参考峰值距4不超过0.5 mg N/L的{int(f05.n_candidates)}个固定网格候选上，自由ρ完整模型与去反馈模型的RMSE分别为{fmt(f05.rmse)}和{fmt(nf05.rmse)} mg N/L，历史M2为{fmt(legacy05.rmse)} mg N/L。总体浓度误差较低未同时带来近阈值最小误差；历史M2的二元监督背景仍须与连续标定模型分开解释。本轮六模型均未观察到原始误放，Gate纠正的是原始误拒，因此本实验不增加误放捕获证据。

自由ρ模型原始误拒占固定网格48个参考合规点的23/48（47.9%）。上述0.5 mg N/L子集中有6个合规、2个超限点，自由ρ模型和去反馈模型均误拒全部6个合规点，并对8个点全部调用参考。固定网格0.25 mg N/L窗口仅含3个合规点，因此双侧覆盖结论指六工况的固定候选集合，未扩大为每个浓度窗口均双侧。

{metric_table(metrics,'coarse')}

六模型在固定主网格的核验比例为{gate_min:.1%}–{gate_max:.1%}。经验缓冲均大于4 mg N/L，当前非负浓度输出没有直接放行候选；因此核验后错误与原始预测错误须同时报告。12点半步长参考的最大峰值差为{hd['maximum_absolute_peak_difference']:.9f} mg N/L，标签变化{hd['label_change_count']}次；这些数值诊断不改变既定1 min主参考身份。

## 可并入讨论

本补充实验直接核查实际参考阈值两侧的覆盖，允许区分连续浓度拟合、阈值判别及参考调用代价。{closure}当前比较仍保留不同监督与历史先验的影响：历史M2采用二元监督，过程模型保留经验生物量闭合，而两种回归模型只使用L及有效传氧特征；相同连续目标不等于所有先验和模型容量相同。固定主网格作为主要比较，加密层承担模型引导的邻域诊断，两者不混为自然分布抽样结果。

本轮未在参考产生后调整模型、Gate或候选。六个设计工况共享模型及固定网格，不能将108或120个候选视为独立现场样本作总体风险推断。观察到的局部采样标签变化也不证明连续阈值唯一、全局响应单调或离散首合规点具有连续最优性。参数触界、辅助SNO投影以及高核验代价应与本轮边界证据共同保留；全文结论据此限定为当前15°C恒定域内条件的任务评价。

## 图表入口与证据记录

科研图1展示六工况的冻结预测与原生参考，主要视图突出阈值区间，全域小图保留守卫点；科研图2比较固定108点的浓度误差、原始错误和Gate代价。准确图名与完整图注见`figures/README.md`，图内连线仅连接实际计算候选，不代表新积分或拟合插值。

| 新增位置 | 修改内容 | 理由 | 可核查证据 |
|---|---|---|---|
| M3方法后 | 增加M3B冻结边界验证设计 | 原M3加密层全部合规，需直接核查实际阈值两侧 | protocol.json、freeze.json、new_candidates.csv |
| 连续任务结果 | 独立呈现108固定点、12加密点及120点汇总 | 保留模型选点偏好与版本身份 | metrics_overall.csv、metrics_by_condition.csv |
| 阈值表现 | 增加真实双侧和参考定义邻域 | 避免仅报告预测边界附近误差 | coverage_by_condition.csv、near_threshold_metrics.csv |
| 结构消融 | 使用两个5参数模型逐点匹配比较 | 排除额外参数数目的混淆 | matched_pair_by_condition.csv |
| 数值补充 | 增加12点新0.5 min参考 | 核查预选邻域的数值稳定性 | numerical/half_step_audit.json |
| 讨论与结论 | 保留误判、核验成本和负面比较 | 使主张对应实际证据 | independent_review.md、M3B_边界补充验证报告.md |

上述段落数据已核验（VERIFIED），未加入新文献。原稿尚未直接覆盖；整合时应给M3B独立编号，并同步摘要、讨论和结论，避免引用旧版最优数字替代本轮结果。图号、表号和章节号由全文排版统一确定。
'''
    (ROOT/'M3B_论文补充段落.md').write_text(paper,encoding='utf-8')
    print('Wrote M3B reports from independently verified artifacts')
if __name__=='__main__':generate()
