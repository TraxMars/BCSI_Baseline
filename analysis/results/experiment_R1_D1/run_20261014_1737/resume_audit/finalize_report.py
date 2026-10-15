import sys,json,datetime,hashlib,shutil
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('/home/duanxuelong/BCSI_Baseline_BGCE')
sys.path.insert(0,str(ROOT/'analysis'))
import experiment_R1_D1 as a
out=ROOT/'analysis/results/experiment_R1_D1/run_20261014_1737'
assert json.loads((out/'experiment_metadata.json').read_text())['status']=='complete'
plan=json.loads((out/'candidate_plan.json').read_text())['records'];s=json.loads((out/'summary_statistics.json').read_text())
rows=[];cache={}
for r in plan:
 key=(r['fold'],r['patch_index']);parent=Path(r['path']).parent
 if key not in cache:
  with np.load(parent/'image_only_maps.npz') as f:x=f['x3'].reshape(64,-1).astype(np.float64)
  with np.load(out/'strict_fp32_oracle'/f"fold_{r['fold']}"/f"patch_{r['patch_index']}"/'oracle_gradient.npz') as f:g=f['gradient'].reshape(64,-1).astype(np.float64)
  cache[key]=(x,g)
 x,g=cache[key]
 with np.load(r['path']) as f:
  idx=np.flatnonzero(f['mask']);d=f['modified_selected'].astype(np.float64)-x[:,idx]
 dot=np.sum(g[:,idx]*d,axis=0)
 cost=float(dot[dot>0].sum());benefit=float(-dot[dot<0].sum())
 rows.append(dict(case_id=r['case_id'],patch_index=r['patch_index'],method=r['method'],repetition=r['repetition'],
  strict_predicted_loss_increase_components=cost,strict_predicted_loss_decrease_components=benefit,
  strict_net_directional_derivative=cost-benefit,oracle_precision='strict_fp32'))
a.r0.write_csv(out/'directional_contribution_decomposition.csv',rows)
cases,fields=a.group_case(rows,['method']);a.r0.write_csv(out/'directional_contribution_case.csv',cases)
extra={}
for method in sorted({r['method'] for r in cases}):
 rr=[r for r in cases if r['method']==method]
 extra[method]={k:a.s.summarize([r[k] for r in rr]) for k in fields}
a.r0.write_json(out/'directional_contribution_summary.json',dict(statistical_unit='case',oracle_precision='strict_fp32',methods=extra))
def stat(table,method,metric,**keys):
 return next(r['metrics'][metric] for r in s[table] if r['group']==dict(method=method,**keys))
methods=['true_boundary','nonboundary','multi','multi_energy_matched','local','local_energy_matched']
lines=['## 最终诊断结论与下一步优先级','',
 '实验已完整完成：8病例×3patch，768条候选，所有48个Phase B阶段任务完成。未重跑Phase A；True Boundary在本报告中即Global Two-Prototype。所有分割指标来自原TF32封存预测，Oracle方向及同精度一阶/有限差分验证来自严格FP32。','',
 '|固定方法|高D纯侧匹配准确率|历史TF32 Boundary收益均值 [病例级95%CI]|Boundary改善病例|净纠正均值 [CI]|净纠正为正病例|',
 '|---|---:|---|---:|---|---:|']
for m in methods:
 acc=stat('semantic',m,'accuracy',domain='high_D')['mean']*100
 bg=stat('correction',m,'boundary_ce_gain');net=stat('correction',m,'net_correction')
 lines.append(f"|{m}|{acc:.2f}%|{bg['mean']:.8f} [{bg['ci95'][0]:.8f}, {bg['ci95'][1]:.8f}]|{bg['positive_cases']}/8|{net['mean']:.3f} [{net['ci95'][0]:.3f}, {net['ci95'][1]:.3f}]|{net['positive_cases']}/8|")
lines += ['',
 '**1. 匹配准确性不够支撑当前无条件插值。** True准确率65.67%，前景侧82.73%、背景侧56.15%，表现出对背景侧较差的匹配。Q均值0.8629且72.77%的候选Q饱和于1，但Q正确性AUROC仅0.5533。Q有弱的排序信息：四档准确率约53.07%、59.66%、64.16%、67.55%；不能把它当作高可信正确概率。高D域31.08%的混合GT网格未用于类别准确率，但完整保留在损失、纠正误伤和梯度分析中。',
 '当匹配与Teacher伪标签冲突，在纯侧网格上Donor正确率只有25.38%，Teacher为74.62%。因此不支持按原型类别覆盖Teacher。A/B/C/D类型中的Teacher正确仅指网格多数伪类别与纯侧GT类别相同，不能推断该网格全部64个原图体素都正确；逐体素Corrected/Harmed另行计算。',
 '**2. 残差方向是优先改进对象。** True高D有效残差位置46.43%与下降方向一致、53.57%相反。严格FP32一阶预测Delta L=+0.00049310，严格FP32实际+0.00050349，历史TF32实际+0.00050505，均支持当前方向整体增加Boundary CE。True在24patch的一阶与严格FP32实际变化符号全部一致。True优于Reverse（配对Boundary收益差+0.00133467、8/8病例同向）仍不代表它优于No Correction。',
 '**3. Non-Boundary的CE改善主要得到方向导数证据支持，不是更好的类别匹配证据。** Non-Boundary匹配准确率58.86%，低于True65.67%；严格FP32一阶Delta L=-0.00054905、同精度实际-0.00053033，历史TF32实际-0.00053428。等能量对照的历史Boundary收益仍为+0.00053016，说明差异不能仅由总扰动大小解释。其平均净纠正-16.29、Dice变化-0.00004821，仍不是有效分割修正。概率校准等因素没有被独立分解，不能唯一归因。',
 '**方向统计的关键区别：** Non-Boundary的A>0比例46.29%并没有高于True46.43%；不能仅依据角度比例声称它整体更好。整体导数按每位置梯度与残差幅值加权；下面的正/负方向贡献分解显示，损失上升项和下降项相抵后的净值才对应一阶损失变化。','',
 '|方法|严格FP32预测损失上升项之和（病例均值）|预测损失下降项幅度之和|净Delta L|',
 '|---|---:|---:|---:|']
for m in ['true_boundary','opposite_match','nonboundary','multi','local']:
 z=extra[m]
 lines.append('|'+m+'|'+ '|'.join(f"{z[k]['mean']:.8f}" for k in ['strict_predicted_loss_increase_components','strict_predicted_loss_decrease_components','strict_net_directional_derivative'])+'|')
lines += ['',
 '**4. 局部表示提供相对改善，尚未建立稳定的实际修正价值。** Multi与Local NN的匹配准确率为66.55%和66.29%，相对True的配对准确率增量CI都跨0。它们相对True的历史Boundary收益增量分别为+0.00055054 [0.00031493, 0.00079863]与+0.00059376 [0.00010750, 0.00105692]；等能量对照仍保持相对改善。',
 '但与No Correction相比，Multi/Local NN的Boundary收益CI和净纠正CI均跨0，只有2/8病例净纠正为正。Multi净纠正均值+0.583但中位数-3.167；Local NN均值+0.125、中位数-1.833，不能忽略正均值由少数病例拉高。各策略Full CE和Dice相对不修正的区间也缺少稳定正收益。因此“比有害两原型更好”不等于“已经能有效补偿Receiver”。',
 '**5. 明确建议：优先研究修正方向与是否需要修正的拒绝条件，暂停直接原型插值的训练升级。** 选择机制判断E：语义可靠性不足和损失方向冲突都有直接证据；局部表示改善了相对损失，但并未稳定改善语义或净纠正，尚不足以把失败单独归因于全局均值压缩。若继续，应把局部表示作为分析工具，在独立预注册控制中研究Receiver需求/方向；当前不建议继续仅增加原型数、搜索Q/alpha，或进入完整Mean Teacher训练。',
 '**数值与保护验证：** 固定两patch、两方向严格FP32 t=.01中心差分4/4通过原容差；t=.001为3/4，通过和失败全部保存。历史TF32两个t在恢复验证中均0/4通过，原失败记录另保留。14项单元测试通过；模型205个参数/BN缓冲张量严格不变；9315个历史文件SHA256一致；768候选、原Phase A封存未变。没有更新权重、BN、EMA或训练。严格FP32以历史Encoder特征为固定输入，仅冻结Decoder重放更换算术精度。',
 '全部8病例已参与原模型训练，LOCO仅隔离当前Donor构造和Receiver GT；10,000次病例Bootstrap不能消除训练复用、重叠开发集、小样本和多配置探索偏差。结果是冻结模型机制诊断，不是未见病例或真实无标签训练收益。','',
 '新增复核：directional_contribution_decomposition.csv / directional_contribution_case.csv / directional_contribution_summary.json；mechanism_summary.png。下面保留完整方法、Q分组、配对比较、全部病例与Q1–Q10报告。','']
audit=out/'resume_audit'/'report_finalization';audit.mkdir(exist_ok=False)
old=(out/'experiment_report.md').read_text();shutil.copy2(out/'experiment_report.md',audit/'generated_experiment_report.md');shutil.copy2(out/'mechanism_diagnosis.md',audit/'generated_mechanism_diagnosis.md')
oldhash=a.r0.sha256(out/'experiment_report.md')
(out/'experiment_report.md').write_text(old.splitlines()[0]+'\n\n'+'\n'.join(lines)+'\n'+'\n'.join(old.splitlines()[1:])+'\n')
(out/'mechanism_diagnosis.md').write_text('# R1-D1最终机制诊断\n\n'+'\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
raw=a.read_csv(out/'case_level_metrics.csv')
fig,axes=plt.subplots(1,3,figsize=(16,5))
labels=['Global 2','Non-boundary','Multi 4/class','Local NN']
for i,m in enumerate(['true_boundary','nonboundary','multi','local']):
 vals=[float(r['boundary_ce_gain']) for r in raw if r['table']=='correction' and r['method']==m]
 axes[0].scatter(np.full(len(vals),i),vals,s=25);axes[0].plot(i,np.mean(vals),'kx',markersize=10)
axes[0].axhline(0,color='k',lw=.7);axes[0].set_xticks(range(4));axes[0].set_xticklabels(labels,rotation=20);axes[0].set_ylabel('Historical TF32 Boundary CE gain');axes[0].set_title('8 equally weighted case means')
rr=a.read_csv(out/'directional_derivative_analysis.csv')
rr=[r for r in rr if r['method'] in ['true_boundary','nonboundary','multi','local']]
x=np.array([float(r['predicted_loss_change']) for r in rr]);y=np.array([float(r['actual_loss_change']) for r in rr]);axes[1].scatter(x,y,s=10);lo=min(x.min(),y.min());hi=max(x.max(),y.max());axes[1].plot([lo,hi],[lo,hi],'k--');axes[1].set_xlabel('Strict FP32 first-order delta loss');axes[1].set_ylabel('Strict FP32 actual delta loss');axes[1].set_title('Same-precision comparison')
q=[stat('q','true_boundary','accuracy',group=str(i))['mean'] for i in range(4)];axes[2].bar(range(4),q);axes[2].set_ylim(0,1);axes[2].set_xticks(range(4));axes[2].set_xticklabels(['0-.25','.25-.5','.5-.75','.75-1']);axes[2].set_ylabel('Pure-grid matching accuracy');axes[2].set_title('Q is not calibrated probability')
fig.tight_layout();fig.savefig(out/'mechanism_summary.png',dpi=150);plt.close(fig)
a.r0.write_json(audit/'report_finalization.json',dict(created_utc=a.now(),scope='post-completion read-only decomposition and report interpretation; no candidate/model/GT selection changes',
 report_sha256_before=oldhash,report_sha256_after=a.r0.sha256(out/'experiment_report.md'),source_sha256=a.r0.sha256(Path(__file__)),primary_summary_sha256=a.r0.sha256(out/'summary_statistics.json')))
print('Final report and direction decomposition written; all primary results unchanged.')
