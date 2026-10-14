# Experiment R1-S0：冻结 Donor–Receiver 边界特征传递可行性

## 研究范围与协议

使用原始 Baseline EMA iter27000，8个已参与有标签训练病例、24个历史固定 patch。LOCO 仅隔离当前 Donor 构造和匹配中的 Receiver GT；这不是未见病例的独立泛化验证。所有预测在 eval、冻结参数及 no_grad 下完成，修改的只有原 Decoder 输入 x3 skip；x4/x5 仍来自原始 Encoder。
主方案固定 Dxyz→Max Pooling→Top2%，次要方案 Trilinear→Top2%；alpha=0.02/0.05、Q=clip(|cos_fg−cos_bg|/0.2)、每位置相对L2上限0.10。Donor原型由每病例边界纯度≥0.75且网格最小边界距离≤3voxel的样本均值，再7病例等权平均。非边界为最小距离>3voxel；count-matched boundary 与 nonboundary 使用逐病例逐类别相同样本数。
匹配尺度统计来自7个Donor全x3网格的等病例均值/二阶矩。余弦仅在标准化空间计算，残差仍在原始特征空间。Receiver自身原型只使用伪标签。无效原型、异常范数、Q≤0.001的格点不强制修正。
原协议的标签交换在竞争匹配+绝对margin下数学等价，不能识别语义作用。保留等价检查；经用户明确授权增加反向匹配（改选另一类原型并保留Q）。随机Mask及随机前景/背景目标各20次、独立CPU Generator；全部结果保留。等能量对照以GT无关标量及相同逐位置cap匹配True residual总L2，不能匹配时明确记录。
每折所有策略、3patch预测及配置先保存并SHA256封存，才解锁该Receiver GT。跨折同病例可换角色。原始两类FP32 logits无损保存，二值预测采用GPU原始softmax>0.5的封存位图；Phase B CPU重放softmax/原CE+Dice，数值容差单独测试。Random先patch内20次平均，再case内3patch平均，最后8病例等权、病例级Bootstrap10000次。

## 主要定量结果

Gain_B=BoundaryCE(无修正)−BoundaryCE(修正)，正值有益。Net=纠正错误−破坏正确，正值有益。下表CI为病例级探索性95%区间。

|映射|alpha|方法|Boundary CE收益均值 [CI]|正收益病例|净纠正均值|Full CE收益|Dice变化|
|---|---:|---|---|---:|---:|---:|---:|
|max|0.02|true_boundary|-0.000505046 [-0.000948663, -2.79451e-05]|1/8|-4.625|-1.90521e-05|-1.3447e-05|
|max|0.02|opposite_match|-0.00183972 [-0.00265021, -0.000895748]|1/8|19.7917|-4.14459e-05|6.25602e-05|
|max|0.02|nonboundary|0.00053428 [0.000114917, 0.00104906]|6/8|-16.2917|1.62637e-05|-4.82129e-05|
|max|0.02|boundary_count_matched|-0.000506919 [-0.00095148, -2.82155e-05]|1/8|-4.45833|-1.92858e-05|-1.30914e-05|
|max|0.02|self|-0.000514709 [-0.000910576, -6.83216e-05]|1/8|-4.20833|-1.9408e-05|-1.15416e-05|
|max|0.02|random_mask|-4.22447e-05 [-7.97596e-05, -3.92783e-06]|2/8|-0.16875|-1.89345e-06|-6.85971e-07|
|max|0.02|random_match|-0.00117954 [-0.00179487, -0.000484057]|1/8|8.28125|-2.98737e-05|2.67421e-05|
|max|0.05|true_boundary|-0.00130106 [-0.00239146, -9.24141e-05]|1/8|-8|-5.20459e-05|-2.24317e-05|
|max|0.05|opposite_match|-0.00462734 [-0.00662924, -0.00228606]|1/8|53.1667|-0.000113192|0.000166685|
|max|0.05|nonboundary|0.0012744 [0.000225947, 0.00257068]|6/8|-40.125|3.58019e-05|-0.000119519|
|max|0.05|boundary_count_matched|-0.00130308 [-0.00239769, -9.29945e-05]|1/8|-8.125|-5.20041e-05|-2.30573e-05|
|max|0.05|self|-0.00131366 [-0.00230248, -0.000188096]|1/8|-6.83333|-5.14746e-05|-1.77524e-05|
|max|0.05|random_mask|-0.000107281 [-0.000199427, -1.25502e-05]|2/8|-0.0270833|-4.99157e-06|-1.26442e-07|
|max|0.05|random_match|-0.0029897 [-0.00451703, -0.00126229]|1/8|21.8042|-8.08623e-05|6.90059e-05|
|trilinear|0.02|true_boundary|-0.000167972 [-0.000822362, 0.000375211]|4/8|-4.75|-2.10192e-06|-1.49038e-05|
|trilinear|0.02|opposite_match|-0.00242923 [-0.00403015, -0.00104483]|0/8|16.5|-8.28399e-05|4.22912e-05|
|trilinear|0.02|nonboundary|0.00121318 [0.000816585, 0.00156606]|8/8|-12.5|5.60123e-05|-3.55248e-05|
|trilinear|0.02|boundary_count_matched|-0.000167949 [-0.000819928, 0.000371849]|4/8|-4.33333|-2.11271e-06|-1.34701e-05|
|trilinear|0.02|self|-0.00016559 [-0.000642656, 0.000282102]|3/8|-3.16667|1.05907e-06|-9.83876e-06|
|trilinear|0.02|random_mask|-4.22447e-05 [-7.97596e-05, -3.92783e-06]|2/8|-0.16875|-1.89345e-06|-6.85971e-07|
|trilinear|0.02|random_match|-0.00130655 [-0.00237501, -0.000426899]|1/8|7.21458|-4.27275e-05|1.84885e-05|
|trilinear|0.05|true_boundary|-0.000494276 [-0.00212833, 0.000866442]|4/8|-6.75|-1.0518e-05|-2.34712e-05|
|trilinear|0.05|opposite_match|-0.00629113 [-0.0103855, -0.0027723]|0/8|39.625|-0.000228779|0.00010071|
|trilinear|0.05|nonboundary|0.00296157 [0.00197577, 0.00384201]|8/8|-27.75|0.000134818|-7.80359e-05|
|trilinear|0.05|boundary_count_matched|-0.000504965 [-0.0021454, 0.00086262]|4/8|-6.79167|-1.12637e-05|-2.34769e-05|
|trilinear|0.05|self|-0.00049394 [-0.00169553, 0.000630017]|3/8|-3.08333|-3.07934e-06|-1.0373e-05|
|trilinear|0.05|random_mask|-0.000107281 [-0.000199427, -1.25502e-05]|2/8|-0.0270833|-4.99157e-06|-1.26442e-07|
|trilinear|0.05|random_match|-0.00339982 [-0.00609964, -0.00117259]|1/8|16.4354|-0.000118355|4.01729e-05|

## 配对比较与扰动公平性

正差值表示True边界优于对照；同时审查原始和等能量对照。

|映射|alpha|对照|Boundary CE优势均值 [CI]|优势病例|
|---|---:|---|---|---:|
|max|0.02|boundary_count_matched|1.87243e-06 [-2.33045e-06,6.17136e-06]|5/8|
|max|0.02|boundary_count_matched_energy_matched|1.15265e-06 [-3.06752e-06,5.4685e-06]|5/8|
|max|0.02|no_correction|-0.000505046 [-0.000948663,-2.79451e-05]|1/8|
|max|0.02|nonboundary|-0.00103933 [-0.00146525,-0.000636669]|0/8|
|max|0.02|nonboundary_energy_matched|-0.00103521 [-0.00148144,-0.000623769]|0/8|
|max|0.02|opposite_match|0.00133467 [0.000797131,0.00181477]|8/8|
|max|0.02|opposite_match_energy_matched|0.00108437 [0.000666597,0.00147582]|8/8|
|max|0.02|random_mask|-0.000462801 [-0.000893925,1.31713e-05]|1/8|
|max|0.02|random_mask_energy_matched|-0.000464829 [-0.000897003,9.15032e-06]|1/8|
|max|0.02|random_match|0.000674497 [0.000404111,0.000914068]|8/8|
|max|0.02|random_match_energy_matched|0.000588377 [0.000364024,0.0007938]|8/8|
|max|0.02|self|9.66265e-06 [-7.16285e-05,8.40888e-05]|5/8|
|max|0.02|self_energy_matched|3.2352e-05 [-5.2755e-05,0.000116126]|5/8|
|max|0.05|boundary_count_matched|2.01788e-06 [-4.83023e-06,9.76865e-06]|4/8|
|max|0.05|boundary_count_matched_energy_matched|9.63092e-07 [-5.7371e-06,8.62315e-06]|4/8|
|max|0.05|no_correction|-0.00130106 [-0.00239146,-9.24141e-05]|1/8|
|max|0.05|nonboundary|-0.00257546 [-0.00364579,-0.00156541]|0/8|
|max|0.05|nonboundary_energy_matched|-0.00257689 [-0.00368336,-0.00154798]|0/8|
|max|0.05|opposite_match|0.00332628 [0.00199739,0.00452106]|8/8|
|max|0.05|opposite_match_energy_matched|0.00268344 [0.00164567,0.00365982]|8/8|
|max|0.05|random_mask|-0.00119378 [-0.00226748,2.76221e-07]|1/8|
|max|0.05|random_mask_energy_matched|-0.00119994 [-0.00226797,-1.10737e-05]|1/8|
|max|0.05|random_match|0.00168864 [0.00101391,0.00228418]|8/8|
|max|0.05|random_match_energy_matched|0.0014491 [0.000889193,0.0019643]|8/8|
|max|0.05|self|1.2601e-05 [-0.00018053,0.000200151]|5/8|
|max|0.05|self_energy_matched|6.79764e-05 [-0.000136413,0.000269985]|5/8|
|trilinear|0.02|boundary_count_matched|-2.26928e-08 [-4.30484e-06,4.24985e-06]|3/8|
|trilinear|0.02|boundary_count_matched_energy_matched|9.02191e-07 [-3.10485e-06,5.65118e-06]|4/8|
|trilinear|0.02|no_correction|-0.000167972 [-0.000822362,0.000375211]|4/8|
|trilinear|0.02|nonboundary|-0.00138116 [-0.00175419,-0.00103814]|0/8|
|trilinear|0.02|nonboundary_energy_matched|-0.00135342 [-0.00173753,-0.000990611]|0/8|
|trilinear|0.02|opposite_match|0.00226125 [0.00119232,0.00343741]|8/8|
|trilinear|0.02|opposite_match_energy_matched|0.00187861 [0.00103415,0.00280265]|8/8|
|trilinear|0.02|random_mask|-0.000125727 [-0.000756192,0.0004037]|4/8|
|trilinear|0.02|random_mask_energy_matched|-0.000130618 [-0.000764534,0.000402192]|4/8|
|trilinear|0.02|random_match|0.00113857 [0.000610822,0.00172333]|8/8|
|trilinear|0.02|random_match_energy_matched|0.0010197 [0.000564353,0.00151847]|8/8|
|trilinear|0.02|self|-2.38225e-06 [-0.000217027,0.000171855]|6/8|
|trilinear|0.02|self_energy_matched|1.64654e-05 [-0.000188862,0.000190429]|6/8|
|trilinear|0.05|boundary_count_matched|1.06892e-05 [2.89184e-06,1.8138e-05]|6/8|
|trilinear|0.05|boundary_count_matched_energy_matched|1.04203e-05 [2.53617e-06,1.77703e-05]|6/8|
|trilinear|0.05|no_correction|-0.000494276 [-0.00212833,0.000866442]|4/8|
|trilinear|0.05|nonboundary|-0.00345585 [-0.00439324,-0.00259688]|0/8|
|trilinear|0.05|nonboundary_energy_matched|-0.00338904 [-0.00435147,-0.00248976]|0/8|
|trilinear|0.05|opposite_match|0.00579686 [0.00304299,0.00882716]|8/8|
|trilinear|0.05|opposite_match_energy_matched|0.00476736 [0.00259315,0.00713828]|8/8|
|trilinear|0.05|random_mask|-0.000386995 [-0.00195993,0.000942497]|4/8|
|trilinear|0.05|random_mask_energy_matched|-0.000401092 [-0.00198294,0.00092991]|4/8|
|trilinear|0.05|random_match|0.00290555 [0.00155441,0.00439622]|8/8|
|trilinear|0.05|random_match_energy_matched|0.00258066 [0.00142051,0.00385036]|8/8|
|trilinear|0.05|self|-3.35363e-07 [-0.000533888,0.000435051]|6/8|
|trilinear|0.05|self_energy_matched|4.26287e-05 [-0.000465317,0.000474021]|6/8|

## 全部Receiver及失败结果

|病例|alpha|Max True收益|纠正|误伤|净纠正|远FP变化|核心远FP变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|06SR5RBREL16DQ6M8LWS|0.02|-0.00158303|54.3333|26.3333|28|-15.3333|-0.333333|
|06SR5RBREL16DQ6M8LWS|0.05|-0.00392666|141|64|77|-41|-1|
|0RZDK210BSMWAA6467LU|0.02|-0.000435699|19.3333|31.3333|-12|0|0|
|0RZDK210BSMWAA6467LU|0.05|-0.00118234|52|71|-19|0|0|
|1D7CUD1955YZPGK8XHJX|0.02|-0.000937259|14.3333|33|-18.6667|-0.666667|0|
|1D7CUD1955YZPGK8XHJX|0.05|-0.00240992|39.6667|93.6667|-54|-1|0|
|1GU15S0GJ6PFNARO469W|0.02|-0.000203533|22.3333|22.6667|-0.333333|0|0|
|1GU15S0GJ6PFNARO469W|0.05|-0.000548842|54|52.6667|1.33333|0|0|
|1MHBF3G6DCPWHSKG7XCP|0.02|-0.00102856|19.3333|28|-8.66667|0|0|
|1MHBF3G6DCPWHSKG7XCP|0.05|-0.00265916|45.3333|78.3333|-33|0|0|
|23X6SY44VT9KFHR7S7OC|0.02|-0.000235357|19|33.3333|-14.3333|1.33333|0|
|23X6SY44VT9KFHR7S7OC|0.05|-0.000591366|46.3333|76.6667|-30.3333|7.66667|0|
|2XL5HSFSE93RMOJDRGR4|0.02|0.000818575|24.6667|27.3333|-2.66667|-4.66667|0|
|2XL5HSFSE93RMOJDRGR4|0.05|0.00201932|70.6667|59.3333|11.3333|-16|0|
|38CWS74285MFGZZXR09Z|0.02|-0.000435506|17.6667|26|-8.33333|-1|0|
|38CWS74285MFGZZXR09Z|0.05|-0.00110949|40|57.3333|-17.3333|-2.33333|0|

## 科学问题Q1–Q10

alpha=0.02：Max True纠正均值23.875、误伤28.5；Mask内误伤16.6667、Mask外误伤11.8333；纯侧网格匹配正确率0.656706；匹配与伪标签冲突位置原错误率0.281595。
alpha=0.05：Max True纠正均值61.125、误伤69.125；Mask内误伤40.9167、Mask外误伤28.2083；纯侧网格匹配正确率0.656706；匹配与伪标签冲突位置原错误率0.281595。

Q1：风险定位的价值通过True+D Mask与20次Random Mask配对比较判断，不能由R0.5的错误覆盖率直接推导修正价值。具体差值和CI见上表；Mask外变化来自原Decoder空间卷积，已分别统计。
Q2：真实边界知识的增量需同时优于非边界、Self、随机匹配及反向匹配，且等能量比较方向一致。标签交换完全等价，不能作为支持语义知识的证据。count-matched比较控制每Donor每类样本数，仍不能消除原型分布和范数差异。
Q3：余弦margin/Q反映特征原型可分性，不能直接称为正确匹配概率。纯侧GT匹配准确率及与伪标签冲突的实际错误率均仅在Phase B评估。混合网格不强制赋真实前景/背景类别。
Q4：Q的固定四档[0,.25)、[.25,.5)、[.5,.75)、[.75,1]分别保存原错误量、纠正量及误伤量，见correction_harm_analysis.csv。Q高未必代表Decoder需要相应原型残差。
Q5：全部纠正、误伤和净纠正见上表及每patch/病例CSV；不得只比较Boundary CE。高D但原本正确的区域误伤分母为Mask内原正确体素，Decoder影响范围另报。
Q6：Full CE、监督0.5×(CE+Dice)、Dice/Jaccard、FP/FN均保留；Boundary CE收益若伴随负净纠正或全局退化，不能判断交互有效。
Q7：完整64维未满足预定收益与副作用门槛；按协议停止TopK复杂化，不把D0-A重要性等同可迁移性。
Q8：Max与Trilinear使用预定同一2%预算，未按GT改变主方案。其收益/误伤差异仅作为敏感性分析，不能在本数据上选最好配置并宣称独立性能。
Q9：目前不支持进入完整Mean Teacher训练；正向残差的错误纠正和误伤需要先得到可靠证据。
Q10：若定位能覆盖错误但所有目标方向都无益，更可能是原型表达、竞争匹配或残差方向/置信度失配；若True与Random Mask相近，风险定位未转化为修正收益；若Self/非边界与True相近，则不能归因于跨样本真实边界知识。该区分是机制解释而非因果证明。

## 事实、解释和未证实假设

直接事实是本冻结checkpoint下各预定修正的实际Decoder重放损失、纠正与误伤及其跨病例配对差异。风险高不等于需要替换为Donor均值；候选库的边界纯侧筛选也不保证局部形态对应。全局两类均值可能丢失空间结构，这是合理解释，但本实验未单独证明。
所有Receiver已被Baseline有标签训练，Donor集合高度重叠；8病例Bootstrap不能消除训练复用、选择checkpoint和小样本偏倚。patch裁剪使距离>20voxel的远FP定义敏感，保守核心与裁剪敏感部分单独保存。4³网格不是x3实际感受野。不能宣称提高整例/未见病例Dice或已实现半监督知识迁移。

## 工程与保护验证

{
  "weights_and_buffers_exactly_unchanged": true,
  "state_tensor_count": 205,
  "all_parameters_frozen": true,
  "no_model_parameter_gradients": true,
  "all_modules_eval": true,
  "protected_file_sha256_unchanged": true,
  "protected_file_count": 673,
  "all_receiver_GT_reads_after_fold_seal": true,
  "no_test_GT_access": true,
  "no_training_or_backward_or_optimizer_or_EMA": true,
  "no_BGS_gain": true,
  "exact_historical_risk_and_masks": true,
  "forward_replay_exact": true,
  "other_encoder_features_unchanged": true,
  "label_swap_invariant": true,
  "opposite_match_authorized": true,
  "random_repetitions_all_saved": true,
  "no_nonfinite_outputs": true,
  "bootstrap_unit": "case",
  "all_8_cases_24patches": true,
  "patch_checks": [
    {
      "case_id": "06SR5RBREL16DQ6M8LWS",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1203,
        2292
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "06SR5RBREL16DQ6M8LWS",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1155,
        2153
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "06SR5RBREL16DQ6M8LWS",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1079,
        2050
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "0RZDK210BSMWAA6467LU",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1383,
        2134
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "0RZDK210BSMWAA6467LU",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1449,
        2470
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "0RZDK210BSMWAA6467LU",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        964,
        1699
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1D7CUD1955YZPGK8XHJX",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        625,
        1240
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1D7CUD1955YZPGK8XHJX",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1491,
        2232
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1D7CUD1955YZPGK8XHJX",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1247,
        1679
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1GU15S0GJ6PFNARO469W",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1158,
        1974
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1GU15S0GJ6PFNARO469W",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1117,
        1971
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1GU15S0GJ6PFNARO469W",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1092,
        1904
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1MHBF3G6DCPWHSKG7XCP",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1138,
        1904
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1MHBF3G6DCPWHSKG7XCP",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1115,
        1964
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "1MHBF3G6DCPWHSKG7XCP",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1097,
        1806
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "23X6SY44VT9KFHR7S7OC",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        936,
        1698
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "23X6SY44VT9KFHR7S7OC",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1388,
        1924
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "23X6SY44VT9KFHR7S7OC",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1205,
        2039
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "2XL5HSFSE93RMOJDRGR4",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1354,
        2084
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "2XL5HSFSE93RMOJDRGR4",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        897,
        1661
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "2XL5HSFSE93RMOJDRGR4",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1345,
        1679
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "38CWS74285MFGZZXR09Z",
      "patch_index": 0,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1172,
        1918
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "38CWS74285MFGZZXR09Z",
      "patch_index": 1,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        1272,
        2132
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    },
    {
      "case_id": "38CWS74285MFGZZXR09Z",
      "patch_index": 2,
      "forward_replay_exact": true,
      "all_other_features_unchanged": true,
      "historical_probability_and_D_and_masks_exact": true,
      "x3_shape": [
        1,
        64,
        28,
        28,
        20
      ],
      "label_swap_invariant": true,
      "self_candidate_counts": [
        904,
        1615
      ],
      "nonfinite_outputs": false,
      "cpu_gpu_probability_max_difference": 1.1920928955078125e-07
    }
  ]
}

## 可复核文件

本目录包含全部原始随机重复、每patch/病例指标、配对CI、原型与匹配图、全FP32预测、逐折封存SHA、数据访问日志、预定病例0/3中间切片可视化和测试记录。原有源码、checkpoint、历史结果的SHA256前后相同；新文件尚不代表已推送GitHub。
