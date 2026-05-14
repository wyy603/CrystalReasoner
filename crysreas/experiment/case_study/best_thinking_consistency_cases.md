# Thinking trace 与最终 CIF 一致性最佳样本

## response 列格式规则

我抽查了前 5 个样本的 `response`，其格式基本一致：先输出 `## Material Report`，其中 `### Crystal Structure` 包含空间群、各元素等价 site 数量与每个 site 的原子数、局部配位和 X-Y 键长；随后 `### Stability` 给出 hull energy 与 formation energy；`### Electronic Properties` 给出 band gap 判定和 Fermi energy；最后用 `<CIF>...</CIF>` 包住最终生成的简化 CIF。

脚本把这些自然语言字段硬编码为正则解析规则：`space group ... (id N)`、`X has n sites: one site has m atoms`、`volume V`、`formation energy per atom is E`、`Fermi energy ... of E eV`，以及包含 `X-Y bond length(s)/distance(s)` 的句子。对于 `shorter/longer` 和 `range` 的键长描述，脚本会保留加权数值或区间端点，再计算预测均值。

最终 CIF 用 ASE 解析晶格、用 spglib 识别空间群和等价原子，体积直接由晶格行列式计算。X-Y 真实键长由周期镜像距离得到，取该元素对首近邻 shell 的平均值与 thinking trace 的 X-Y 预测均值比较。

注意：当前 `conditional+thinking.parquet` 与最终 CIF 不包含真实 Fermi energy 标签，因此 Fermi energy 只能记录 thinking trace 的预测值，不能在这个数据源内计算真实误差。formation energy 的真实值来自 prompt。

## 前 5 个样本抽查摘要

| row | mp_id | SG预测/真实 | 体积预测/真实 | site是否匹配 | 键长pair数 | Fermi预测 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | mp-756729 | 166/166 | 150.91/148.917724 | True | 3 | 3.36 |
| 17 | mp-756729 | 166/12 | 159.55/151.162811 | False | 3 | 2.96 |
| 18 | mp-756729 | 166/160 | 153.91/150.389741 | False | 3 | 3.44 |
| 19 | mp-756729 | 166/1 | 155.89/150.707224 | False | 3 | None |
| 20 | mp-756729 | 166/8 | 152.46/154.190923 | False | 3 | None |

## 选出的 3 个最佳样本

### 样本 1: `mp-6613` / row `621`

- 总分：`99.950908`
- 空间群：预测 `Fm-3m` / `225`；最终 CIF `Fm-3m` / `225`；匹配 `True`
- 体积：预测 `153.87`；最终 CIF `153.875907`；相对误差 `0.003839%`
- site：预测 `{'Ba': [2], 'O': [6], 'Ta': [1], 'Y': [1]}`；最终 CIF `{'Ba': [2], 'O': [6], 'Ta': [1], 'Y': [1]}`；匹配 `True`
- formation energy：预测 `-3.566` eV/atom；真实 prompt `-3.5658` eV/atom；误差 `0.0002`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Ba-O | 3.01 | 3.010186 | 0.000186 | True |
| O-Ta | 2.0 | 1.998288 | 0.001712 | True |
| O-Y | 2.25 | 2.25489 | 0.00489 | True |

最终简化 CIF：

```text
P1
6.01490180 6.01490180 6.01490180
60 60 60
Ba 1 0.25000000 0.25000000 0.25000000
Ba 1 0.75000000 0.75000000 0.75000000
Y 1 0.50000000 0.50000000 0.50000000
Ta 1 0.00000000 0.00000000 0.00000000
O 1 0.76508300 0.23491700 0.23491700
O 1 0.23491700 0.76508300 0.76508300
O 1 0.23491700 0.76508300 0.23491700
O 1 0.76508300 0.23491700 0.76508300
O 1 0.23491700 0.23491700 0.76508300
O 1 0.76508300 0.76508300 0.23491700
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_Ba2YTaO6
_symmetry_space_group_name_H-M   Fm-3m
_cell_length_a   8.50635570
_cell_length_b   8.50635570
_cell_length_c   8.50635570
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   225
_chemical_formula_structural   Ba2YTaO6
_chemical_formula_sum   'Ba8 Y4 Ta4 O24'
_cell_volume   615.50362872
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-y, x, z'
  4  'y, -x, -z'
  5  '-x, -y, z'
  6  'x, y, -z'
  7  'y, -x, z'
  8  '-y, x, -z'
  9  'x, -y, -z'
  10  '-x, y, z'
  11  '-y, -x, -z'
  12  'y, x, z'
  13  '-x, y, -z'
  14  'x, -y, z'
  15  'y, x, -z'
  16  '-y, -x, z'
  17  'z, x, y'
  18  '-z, -x, -y'
  19  'z, -y, x'
  20  '-z, y, -x'
  21  'z, -x, -y'
  22  '-z, x, y'
  23  'z, y, -x'
  24  '-z, -y, x'
  25  '-z, x, -y'
  26  'z, -x, y'
  27  '-z, -y, -x'
  28  'z, y, x'
  29  '-z, -x, y'
  30  'z, x, -y'
  31  '-z, y, x'
  32  'z, -y, -x'
  33  'y, z, x'
  34  '-y, -z, -x'
  35  'x, z, -y'
  36  '-x, -z, y'
  37  '-y, z, -x'
  38  'y, -z, x'
  39  '-x, z, y'
  40  'x, -z, -y'
  41  '-y, -z, x'
  42  'y, z, -x'
  43  '-x, -z, -y'
  44  'x, z, y'
  45  'y, -z, -x'
  46  '-y, z, x'
  47  'x, -z, y'
  48  '-x, z, -y'
  49  'x+1/2, y+1/2, z'
  50  '-x+1/2, -y+1/2, -z'
  51  '-y+1/2, x+1/2, z'
  52  'y+1/2, -x+1/2, -z'
  53  '-x+1/2, -y+1/2, z'
  54  'x+1/2, y+1/2, -z'
  55  'y+1/2, -x+1/2, z'
  56  '-y+1/2, x+1/2, -z'
  57  'x+1/2, -y+1/2, -z'
  58  '-x+1/2, y+1/2, z'
  59  '-y+1/2, -x+1/2, -z'
  60  'y+1/2, x+1/2, z'
  61  '-x+1/2, y+1/2, -z'
  62  'x+1/2, -y+1/2, z'
  63  'y+1/2, x+1/2, -z'
  64  '-y+1/2, -x+1/2, z'
  65  'z+1/2, x+1/2, y'
  66  '-z+1/2, -x+1/2, -y'
  67  'z+1/2, -y+1/2, x'
  68  '-z+1/2, y+1/2, -x'
  69  'z+1/2, -x+1/2, -y'
  70  '-z+1/2, x+1/2, y'
  71  'z+1/2, y+1/2, -x'
  72  '-z+1/2, -y+1/2, x'
  73  '-z+1/2, x+1/2, -y'
  74  'z+1/2, -x+1/2, y'
  75  '-z+1/2, -y+1/2, -x'
  76  'z+1/2, y+1/2, x'
  77  '-z+1/2, -x+1/2, y'
  78  'z+1/2, x+1/2, -y'
  79  '-z+1/2, y+1/2, x'
  80  'z+1/2, -y+1/2, -x'
  81  'y+1/2, z+1/2, x'
  82  '-y+1/2, -z+1/2, -x'
  83  'x+1/2, z+1/2, -y'
  84  '-x+1/2, -z+1/2, y'
  85  '-y+1/2, z+1/2, -x'
  86  'y+1/2, -z+1/2, x'
  87  '-x+1/2, z+1/2, y'
  88  'x+1/2, -z+1/2, -y'
  89  '-y+1/2, -z+1/2, x'
  90  'y+1/2, z+1/2, -x'
  91  '-x+1/2, -z+1/2, -y'
  92  'x+1/2, z+1/2, y'
  93  'y+1/2, -z+1/2, -x'
  94  '-y+1/2, z+1/2, x'
  95  'x+1/2, -z+1/2, y'
  96  '-x+1/2, z+1/2, -y'
  97  'x+1/2, y, z+1/2'
  98  '-x+1/2, -y, -z+1/2'
  99  '-y+1/2, x, z+1/2'
  100  'y+1/2, -x, -z+1/2'
  101  '-x+1/2, -y, z+1/2'
  102  'x+1/2, y, -z+1/2'
  103  'y+1/2, -x, z+1/2'
  104  '-y+1/2, x, -z+1/2'
  105  'x+1/2, -y, -z+1/2'
  106  '-x+1/2, y, z+1/2'
  107  '-y+1/2, -x, -z+1/2'
  108  'y+1/2, x, z+1/2'
  109  '-x+1/2, y, -z+1/2'
  110  'x+1/2, -y, z+1/2'
  111  'y+1/2, x, -z+1/2'
  112  '-y+1/2, -x, z+1/2'
  113  'z+1/2, x, y+1/2'
  114  '-z+1/2, -x, -y+1/2'
  115  'z+1/2, -y, x+1/2'
  116  '-z+1/2, y, -x+1/2'
  117  'z+1/2, -x, -y+1/2'
  118  '-z+1/2, x, y+1/2'
  119  'z+1/2, y, -x+1/2'
  120  '-z+1/2, -y, x+1/2'
  121  '-z+1/2, x, -y+1/2'
  122  'z+1/2, -x, y+1/2'
  123  '-z+1/2, -y, -x+1/2'
  124  'z+1/2, y, x+1/2'
  125  '-z+1/2, -x, y+1/2'
  126  'z+1/2, x, -y+1/2'
  127  '-z+1/2, y, x+1/2'
  128  'z+1/2, -y, -x+1/2'
  129  'y+1/2, z, x+1/2'
  130  '-y+1/2, -z, -x+1/2'
  131  'x+1/2, z, -y+1/2'
  132  '-x+1/2, -z, y+1/2'
  133  '-y+1/2, z, -x+1/2'
  134  'y+1/2, -z, x+1/2'
  135  '-x+1/2, z, y+1/2'
  136  'x+1/2, -z, -y+1/2'
  137  '-y+1/2, -z, x+1/2'
  138  'y+1/2, z, -x+1/2'
  139  '-x+1/2, -z, -y+1/2'
  140  'x+1/2, z, y+1/2'
  141  'y+1/2, -z, -x+1/2'
  142  '-y+1/2, z, x+1/2'
  143  'x+1/2, -z, y+1/2'
  144  '-x+1/2, z, -y+1/2'
  145  'x, y+1/2, z+1/2'
  146  '-x, -y+1/2, -z+1/2'
  147  '-y, x+1/2, z+1/2'
  148  'y, -x+1/2, -z+1/2'
  149  '-x, -y+1/2, z+1/2'
  150  'x, y+1/2, -z+1/2'
  151  'y, -x+1/2, z+1/2'
  152  '-y, x+1/2, -z+1/2'
  153  'x, -y+1/2, -z+1/2'
  154  '-x, y+1/2, z+1/2'
  155  '-y, -x+1/2, -z+1/2'
  156  'y, x+1/2, z+1/2'
  157  '-x, y+1/2, -z+1/2'
  158  'x, -y+1/2, z+1/2'
  159  'y, x+1/2, -z+1/2'
  160  '-y, -x+1/2, z+1/2'
  161  'z, x+1/2, y+1/2'
  162  '-z, -x+1/2, -y+1/2'
  163  'z, -y+1/2, x+1/2'
  164  '-z, y+1/2, -x+1/2'
  165  'z, -x+1/2, -y+1/2'
  166  '-z, x+1/2, y+1/2'
  167  'z, y+1/2, -x+1/2'
  168  '-z, -y+1/2, x+1/2'
  169  '-z, x+1/2, -y+1/2'
  170  'z, -x+1/2, y+1/2'
  171  '-z, -y+1/2, -x+1/2'
  172  'z, y+1/2, x+1/2'
  173  '-z, -x+1/2, y+1/2'
  174  'z, x+1/2, -y+1/2'
  175  '-z, y+1/2, x+1/2'
  176  'z, -y+1/2, -x+1/2'
  177  'y, z+1/2, x+1/2'
  178  '-y, -z+1/2, -x+1/2'
  179  'x, z+1/2, -y+1/2'
  180  '-x, -z+1/2, y+1/2'
  181  '-y, z+1/2, -x+1/2'
  182  'y, -z+1/2, x+1/2'
  183  '-x, z+1/2, y+1/2'
  184  'x, -z+1/2, -y+1/2'
  185  '-y, -z+1/2, x+1/2'
  186  'y, z+1/2, -x+1/2'
  187  '-x, -z+1/2, -y+1/2'
  188  'x, z+1/2, y+1/2'
  189  'y, -z+1/2, -x+1/2'
  190  '-y, z+1/2, x+1/2'
  191  'x, -z+1/2, y+1/2'
  192  '-x, z+1/2, -y+1/2'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ba  Ba0  8  0.25000000  0.25000000  0.25000000  1
  Y  Y1  4  0.00000000  0.00000000  0.50000000  1
  Ta  Ta2  4  0.00000000  0.00000000  0.00000000  1
  O  O3  24  0.00000000  0.00000000  0.23491700  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.06033694631016458, "sum_of_information": 24.979495772408136, "token_num": 414}` |
| gt | `{"comp": "Ba2 Y1 Ta1 O6", "mp_id": "mp-6613", "spg": 225}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-83.40325927734375` |
| energy_above_hull | `0.0050886022656246865` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
6.04115019 6.04115030 6.04114971
60 60 60
Ba 1 0.25000000 0.25000000 0.25000000
Ba 1 0.75000001 0.74999999 0.74999999
Y 1 0.50000000 0.50000000 0.50000001
Ta 1 0.00000000 0.00000001 -0.00000001
O 1 0.76460919 0.23539082 0.23539082
O 1 0.23539082 0.76460919 0.76460918
O 1 0.23539082 0.76460918 0.23539082
O 1 0.76460919 0.23539082 0.76460918
O 1 0.23539081 0.23539082 0.76460918
O 1 0.76460918 0.76460918 0.23539082
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 2: `mp-1080489` / row `14505`

- 总分：`99.816967`
- 空间群：预测 `P-62m` / `189`；最终 CIF `P-62m` / `189`；匹配 `True`
- 体积：预测 `143.59`；最终 CIF `143.612616`；相对误差 `0.015748%`
- site：预测 `{'Ge': [1, 2], 'Hf': [3], 'Ru': [3]}`；最终 CIF `{'Ge': [1, 2], 'Hf': [3], 'Ru': [3]}`；匹配 `True`
- formation energy：预测 `-0.807` eV/atom；真实 prompt `-0.807` eV/atom；误差 `0.0`
- Fermi energy：预测 `7.48` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Ge-Hf | 2.752 | 2.750437 | 0.001563 | True |
| Ge-Ru | 2.515 | 2.518895 | 0.003895 | True |
| Hf-Ru | 3.016667 | 3.016103 | 0.000564 | True |
| Ru-Ru | 2.8 | 2.827435 | 0.027435 | True |

最终简化 CIF：

```text
P1
6.58356040 6.58356040 3.82595900
90 90 120
Hf 1 0.41799100 0.41799100 0.50000000
Hf 1 0.58200900 0.00000000 0.50000000
Hf 1 0.00000000 0.58200900 0.50000000
Ge 1 0.66666700 0.33333300 0.00000000
Ge 1 0.33333300 0.66666700 0.00000000
Ge 1 0.00000000 0.00000000 0.50000000
Ru 1 0.75204600 0.75204600 0.00000000
Ru 1 0.24795400 0.00000000 0.00000000
Ru 1 0.00000000 0.24795400 0.00000000
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_HfGeRu
_symmetry_space_group_name_H-M   P-62m
_cell_length_a   6.58356040
_cell_length_b   6.58356040
_cell_length_c   3.82595900
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   120.00000000
_symmetry_Int_Tables_number   189
_chemical_formula_structural   HfGeRu
_chemical_formula_sum   'Hf3 Ge3 Ru3'
_cell_volume   143.61261559
_cell_formula_units_Z   3
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x+y, -x, -z'
  3  '-y, x-y, z'
  4  'x, y, -z'
  5  '-x+y, -x, z'
  6  '-y, x-y, -z'
  7  'y, x, z'
  8  '-x, -x+y, -z'
  9  'x-y, -y, z'
  10  'y, x, -z'
  11  '-x, -x+y, z'
  12  'x-y, -y, -z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Hf  Hf0  3  0.00000000  0.58200900  0.50000000  1
  Ge  Ge1  2  0.33333333  0.66666667  0.00000000  1
  Ge  Ge2  1  0.00000000  0.00000000  0.50000000  1
  Ru  Ru3  3  0.00000000  0.24795400  0.00000000  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.14813766807544174, "sum_of_information": 56.4404515367433, "token_num": 381}` |
| gt | `{"comp": "Hf1 Ge1 Ru1", "mp_id": "mp-1080489", "spg": 189}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-77.6430435180664` |
| energy_above_hull | `0.07574263910373169` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
6.76345025 6.76345072 3.73522377
90 90 120
Hf 1 0.41705930 0.41705930 0.50000000
Hf 1 0.58294074 -0.00000004 0.50000000
Hf 1 -0.00000005 0.58294075 0.50000000
Ge 1 0.66666685 0.33333314 -0.00000000
Ge 1 0.33333313 0.66666684 0.00000000
Ge 1 0.00000000 -0.00000000 0.50000000
Ru 1 0.75157317 0.75157317 0.00000000
Ru 1 0.24842678 0.00000007 -0.00000000
Ru 1 0.00000007 0.24842678 -0.00000000
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 3: `mp-14620` / row `3105`

- 总分：`99.713593`
- 空间群：预测 `Fd-3m` / `227`；最终 CIF `Fd-3m` / `227`；匹配 `True`
- 体积：预测 `389.31`；最终 CIF `389.362165`；相对误差 `0.013397%`
- site：预测 `{'Cd': [2], 'Se': [8], 'Tm': [4]}`；最终 CIF `{'Cd': [2], 'Se': [8], 'Tm': [4]}`；匹配 `True`
- formation energy：预测 `-1.767` eV/atom；真实 prompt `-1.7665` eV/atom；误差 `0.0005`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Cd-Se | 2.65 | 2.66716 | 0.01716 | True |
| Se-Tm | 2.82 | 2.809859 | 0.010141 | True |

最终简化 CIF：

```text
P1
8.19639567 8.19639567 8.19639567
60 60 60
Tm 1 0.50000000 0.50000000 0.00000000
Tm 1 0.50000000 0.50000000 0.50000000
Tm 1 0.00000000 0.50000000 0.50000000
Tm 1 0.50000000 0.00000000 0.50000000
Cd 1 0.12500000 0.12500000 0.12500000
Cd 1 0.87500000 0.87500000 0.87500000
Se 1 0.74238400 0.74238400 0.74238400
Se 1 0.25761600 0.25761600 0.72623000
Se 1 0.25761600 0.72623000 0.25761600
Se 1 0.72623000 0.25761600 0.25761600
Se 1 0.74238400 0.27377000 0.74238400
Se 1 0.27377000 0.74238400 0.74238400
Se 1 0.25761600 0.25761600 0.25761600
Se 1 0.74238400 0.74238400 0.27377000
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_Tm2CdSe4
_symmetry_space_group_name_H-M   Fd-3m
_cell_length_a   11.59145392
_cell_length_b   11.59145392
_cell_length_c   11.59145392
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   227
_chemical_formula_structural   Tm2CdSe4
_chemical_formula_sum   'Tm16 Cd8 Se32'
_cell_volume   1557.44865907
_cell_formula_units_Z   8
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-y+1/4, x+1/4, z+1/4'
  3  '-x, -y+1/2, z+1/2'
  4  'y+3/4, -x+1/4, z+3/4'
  5  'x, -y, -z'
  6  '-y+1/4, -x+3/4, -z+3/4'
  7  '-x, y+1/2, -z+1/2'
  8  'y+3/4, x+3/4, -z+1/4'
  9  'z, x, y'
  10  'z+1/4, -y+1/4, x+1/4'
  11  'z+1/2, -x, -y+1/2'
  12  'z+3/4, y+3/4, -x+1/4'
  13  '-z, x, -y'
  14  '-z+3/4, -y+1/4, -x+3/4'
  15  '-z+1/2, -x, y+1/2'
  16  '-z+1/4, y+3/4, x+3/4'
  17  'y, z, x'
  18  'x+1/4, z+1/4, -y+1/4'
  19  '-y+1/2, z+1/2, -x'
  20  '-x+1/4, z+3/4, y+3/4'
  21  '-y, -z, x'
  22  '-x+3/4, -z+3/4, -y+1/4'
  23  'y+1/2, -z+1/2, -x'
  24  'x+3/4, -z+1/4, y+3/4'
  25  '-x+1/4, -y+1/4, -z+1/4'
  26  'y, -x, -z'
  27  'x+1/4, y+3/4, -z+3/4'
  28  '-y+1/2, x, -z+1/2'
  29  '-x+1/4, y+1/4, z+1/4'
  30  'y, x+1/2, z+1/2'
  31  'x+1/4, -y+3/4, z+3/4'
  32  '-y+1/2, -x+1/2, z'
  33  '-z+1/4, -x+1/4, -y+1/4'
  34  '-z, y, -x'
  35  '-z+3/4, x+1/4, y+3/4'
  36  '-z+1/2, -y+1/2, x'
  37  'z+1/4, -x+1/4, y+1/4'
  38  'z+1/2, y, x+1/2'
  39  'z+3/4, x+1/4, -y+3/4'
  40  'z, -y+1/2, -x+1/2'
  41  '-y+1/4, -z+1/4, -x+1/4'
  42  '-x, -z, y'
  43  'y+3/4, -z+3/4, x+1/4'
  44  'x, -z+1/2, -y+1/2'
  45  'y+1/4, z+1/4, -x+1/4'
  46  'x+1/2, z+1/2, y'
  47  '-y+3/4, z+3/4, x+1/4'
  48  '-x+1/2, z, -y+1/2'
  49  'x+1/2, y+1/2, z'
  50  '-y+3/4, x+3/4, z+1/4'
  51  '-x+1/2, -y, z+1/2'
  52  'y+1/4, -x+3/4, z+3/4'
  53  'x+1/2, -y+1/2, -z'
  54  '-y+3/4, -x+1/4, -z+3/4'
  55  '-x+1/2, y, -z+1/2'
  56  'y+1/4, x+1/4, -z+1/4'
  57  'z+1/2, x+1/2, y'
  58  'z+3/4, -y+3/4, x+1/4'
  59  'z, -x+1/2, -y+1/2'
  60  'z+1/4, y+1/4, -x+1/4'
  61  '-z+1/2, x+1/2, -y'
  62  '-z+1/4, -y+3/4, -x+3/4'
  63  '-z, -x+1/2, y+1/2'
  64  '-z+3/4, y+1/4, x+3/4'
  65  'y+1/2, z+1/2, x'
  66  'x+3/4, z+3/4, -y+1/4'
  67  '-y, z, -x'
  68  '-x+3/4, z+1/4, y+3/4'
  69  '-y+1/2, -z+1/2, x'
  70  '-x+1/4, -z+1/4, -y+1/4'
  71  'y, -z, -x'
  72  'x+1/4, -z+3/4, y+3/4'
  73  '-x+3/4, -y+3/4, -z+1/4'
  74  'y+1/2, -x+1/2, -z'
  75  'x+3/4, y+1/4, -z+3/4'
  76  '-y, x+1/2, -z+1/2'
  77  '-x+3/4, y+3/4, z+1/4'
  78  'y+1/2, x, z+1/2'
  79  'x+3/4, -y+1/4, z+3/4'
  80  '-y, -x, z'
  81  '-z+3/4, -x+3/4, -y+1/4'
  82  '-z+1/2, y+1/2, -x'
  83  '-z+1/4, x+3/4, y+3/4'
  84  '-z, -y, x'
  85  'z+3/4, -x+3/4, y+1/4'
  86  'z, y+1/2, x+1/2'
  87  'z+1/4, x+3/4, -y+3/4'
  88  'z+1/2, -y, -x+1/2'
  89  '-y+3/4, -z+3/4, -x+1/4'
  90  '-x+1/2, -z+1/2, y'
  91  'y+1/4, -z+1/4, x+1/4'
  92  'x+1/2, -z, -y+1/2'
  93  'y+3/4, z+3/4, -x+1/4'
  94  'x, z, y'
  95  '-y+1/4, z+1/4, x+1/4'
  96  '-x, z+1/2, -y+1/2'
  97  'x+1/2, y, z+1/2'
  98  '-y+3/4, x+1/4, z+3/4'
  99  '-x+1/2, -y+1/2, z'
  100  'y+1/4, -x+1/4, z+1/4'
  101  'x+1/2, -y, -z+1/2'
  102  '-y+3/4, -x+3/4, -z+1/4'
  103  '-x+1/2, y+1/2, -z'
  104  'y+1/4, x+3/4, -z+3/4'
  105  'z+1/2, x, y+1/2'
  106  'z+3/4, -y+1/4, x+3/4'
  107  'z, -x, -y'
  108  'z+1/4, y+3/4, -x+3/4'
  109  '-z+1/2, x, -y+1/2'
  110  '-z+1/4, -y+1/4, -x+1/4'
  111  '-z, -x, y'
  112  '-z+3/4, y+3/4, x+1/4'
  113  'y+1/2, z, x+1/2'
  114  'x+3/4, z+1/4, -y+3/4'
  115  '-y, z+1/2, -x+1/2'
  116  '-x+3/4, z+3/4, y+1/4'
  117  '-y+1/2, -z, x+1/2'
  118  '-x+1/4, -z+3/4, -y+3/4'
  119  'y, -z+1/2, -x+1/2'
  120  'x+1/4, -z+1/4, y+1/4'
  121  '-x+3/4, -y+1/4, -z+3/4'
  122  'y+1/2, -x, -z+1/2'
  123  'x+3/4, y+3/4, -z+1/4'
  124  '-y, x, -z'
  125  '-x+3/4, y+1/4, z+3/4'
  126  'y+1/2, x+1/2, z'
  127  'x+3/4, -y+3/4, z+1/4'
  128  '-y, -x+1/2, z+1/2'
  129  '-z+3/4, -x+1/4, -y+3/4'
  130  '-z+1/2, y, -x+1/2'
  131  '-z+1/4, x+1/4, y+1/4'
  132  '-z, -y+1/2, x+1/2'
  133  'z+3/4, -x+1/4, y+3/4'
  134  'z, y, x'
  135  'z+1/4, x+1/4, -y+1/4'
  136  'z+1/2, -y+1/2, -x'
  137  '-y+3/4, -z+1/4, -x+3/4'
  138  '-x+1/2, -z, y+1/2'
  139  'y+1/4, -z+3/4, x+3/4'
  140  'x+1/2, -z+1/2, -y'
  141  'y+3/4, z+1/4, -x+3/4'
  142  'x, z+1/2, y+1/2'
  143  '-y+1/4, z+3/4, x+3/4'
  144  '-x, z, -y'
  145  'x, y+1/2, z+1/2'
  146  '-y+1/4, x+3/4, z+3/4'
  147  '-x, -y, z'
  148  'y+3/4, -x+3/4, z+1/4'
  149  'x, -y+1/2, -z+1/2'
  150  '-y+1/4, -x+1/4, -z+1/4'
  151  '-x, y, -z'
  152  'y+3/4, x+1/4, -z+3/4'
  153  'z, x+1/2, y+1/2'
  154  'z+1/4, -y+3/4, x+3/4'
  155  'z+1/2, -x+1/2, -y'
  156  'z+3/4, y+1/4, -x+3/4'
  157  '-z, x+1/2, -y+1/2'
  158  '-z+3/4, -y+3/4, -x+1/4'
  159  '-z+1/2, -x+1/2, y'
  160  '-z+1/4, y+1/4, x+1/4'
  161  'y, z+1/2, x+1/2'
  162  'x+1/4, z+3/4, -y+3/4'
  163  '-y+1/2, z, -x+1/2'
  164  '-x+1/4, z+1/4, y+1/4'
  165  '-y, -z+1/2, x+1/2'
  166  '-x+3/4, -z+1/4, -y+3/4'
  167  'y+1/2, -z, -x+1/2'
  168  'x+3/4, -z+3/4, y+1/4'
  169  '-x+1/4, -y+3/4, -z+3/4'
  170  'y, -x+1/2, -z+1/2'
  171  'x+1/4, y+1/4, -z+1/4'
  172  '-y+1/2, x+1/2, -z'
  173  '-x+1/4, y+3/4, z+3/4'
  174  'y, x, z'
  175  'x+1/4, -y+1/4, z+1/4'
  176  '-y+1/2, -x, z+1/2'
  177  '-z+1/4, -x+3/4, -y+3/4'
  178  '-z, y+1/2, -x+1/2'
  179  '-z+3/4, x+3/4, y+1/4'
  180  '-z+1/2, -y, x+1/2'
  181  'z+1/4, -x+3/4, y+3/4'
  182  'z+1/2, y+1/2, x'
  183  'z+3/4, x+3/4, -y+1/4'
  184  'z, -y, -x'
  185  '-y+1/4, -z+3/4, -x+3/4'
  186  '-x, -z+1/2, y+1/2'
  187  'y+3/4, -z+1/4, x+3/4'
  188  'x, -z, -y'
  189  'y+1/4, z+3/4, -x+3/4'
  190  'x+1/2, z, y+1/2'
  191  '-y+3/4, z+1/4, x+3/4'
  192  '-x+1/2, z+1/2, -y'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Tm  Tm0  16  0.12500000  0.12500000  0.12500000  1
  Cd  Cd1  8  0.00000000  0.00000000  0.50000000  1
  Se  Se2  32  0.11738400  0.11738400  0.88261600  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.07320927549883668, "sum_of_information": 41.43644993234156, "token_num": 566}` |
| gt | `{"comp": "Tm2 Cd1 Se4", "mp_id": "mp-14620", "spg": 227}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-68.674560546875` |
| energy_above_hull | `0.01485339022321508` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `True` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
8.26963337 8.26963294 8.26963316
59.9893 59.9893 59.9893
Tm 1 0.50000000 0.50000000 0.00000000
Tm 1 0.50000000 0.50000000 0.50000000
Tm 1 -0.00000000 0.50000000 0.50000000
Tm 1 0.50000000 0.00000000 0.50000000
Cd 1 0.12498464 0.12498464 0.12498464
Cd 1 0.87501536 0.87501536 0.87501536
Se 1 0.74236164 0.74236164 0.74236164
Se 1 0.25763843 0.25763842 0.72631881
Se 1 0.25763842 0.72631881 0.25763842
Se 1 0.72631881 0.25763842 0.25763842
Se 1 0.74236158 0.27368119 0.74236157
Se 1 0.27368119 0.74236158 0.74236158
Se 1 0.25763836 0.25763835 0.25763836
Se 1 0.74236158 0.74236158 0.27368119
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 4: `mp-1030160` / row `1456`

- 总分：`99.673752`
- 空间群：预测 `P3m1` / `156`；最终 CIF `P3m1` / `156`；匹配 `True`
- 体积：预测 `416.01`；最终 CIF `417.193947`；相对误差 `0.283788%`
- site：预测 `{'Mo': [1, 1, 1], 'Se': [1, 1], 'Te': [1, 1, 1, 1, 1, 1], 'W': [1]}`；最终 CIF `{'Mo': [1, 1, 1], 'Se': [1, 1], 'Te': [1, 1, 1, 1, 1, 1], 'W': [1]}`；匹配 `True`
- formation energy：预测 `-0.335` eV/atom；真实 prompt `-0.3346` eV/atom；误差 `0.0004`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Mo-Te | 2.73 | 2.726535 | 0.003465 | True |
| Se-W | 2.59 | 2.590781 | 0.000781 | True |

最终简化 CIF：

```text
P1
3.49263162 3.49263162 39.49133400
90 90 120
Te 1 0.33333300 0.66666700 0.32841000
Te 1 0.66666700 0.33333300 0.04831600
Te 1 0.66666700 0.33333300 0.42400300
Te 1 0.66666700 0.33333300 0.14106000
Te 1 0.66666700 0.33333300 0.51683600
Te 1 0.33333300 0.66666700 0.23518100
Mo 1 0.33333300 0.66666700 0.09391800
Mo 1 0.33333300 0.66666700 0.46964600
Mo 1 0.66666700 0.33333300 0.28185800
W 1 0.66666700 0.33333300 0.65755500
Se 1 0.33333300 0.66666700 0.69868400
Se 1 0.33333300 0.66666700 0.61630400
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_Te6Mo3WSe2
_symmetry_space_group_name_H-M   P3m1
_cell_length_a   3.49263162
_cell_length_b   3.49263162
_cell_length_c   39.49133400
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   120.00000000
_symmetry_Int_Tables_number   156
_chemical_formula_structural   Te6Mo3WSe2
_chemical_formula_sum   'Te6 Mo3 W1 Se2'
_cell_volume   417.19394726
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-y, x-y, z'
  3  '-x+y, -x, z'
  4  '-y, -x, z'
  5  '-x+y, y, z'
  6  'x, x-y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Te  Te0  1  0.00000000  0.00000000  0.04831600  1
  Te  Te1  1  0.00000000  0.00000000  0.14106000  1
  Te  Te2  1  0.00000000  0.00000000  0.42400300  1
  Te  Te3  1  0.00000000  0.00000000  0.51683600  1
  Te  Te4  1  0.66666667  0.33333333  0.23518100  1
  Te  Te5  1  0.66666667  0.33333333  0.32841000  1
  Mo  Mo6  1  0.00000000  0.00000000  0.28185800  1
  Mo  Mo7  1  0.66666667  0.33333333  0.09391800  1
  Mo  Mo8  1  0.66666667  0.33333333  0.46964600  1
  W  W9  1  0.00000000  0.00000000  0.65755500  1
  Se  Se10  1  0.66666667  0.33333333  0.61630400  1
  Se  Se11  1  0.66666667  0.33333333  0.69868400  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.21963618793140377, "sum_of_information": 107.62173208638785, "token_num": 490}` |
| gt | `{"comp": "Te6 Mo3 W1 Se2", "mp_id": "mp-1030160", "spg": 156}` |
| smact_validity | `False` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `None` |
| energies | `None` |
| energy_above_hull | `None` |
| is_stable | `None` |
| is_novel | `None` |
| is_unique | `None` |
| stable_unique_novel | `None` |

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 5: `mp-1104719` / row `8746`

- 总分：`99.647778`
- 空间群：预测 `I4/mmm` / `139`；最终 CIF `I4/mmm` / `139`；匹配 `True`
- 体积：预测 `197.07`；最终 CIF `197.291761`；相对误差 `0.112402%`
- site：预测 `{'Al': [4, 4], 'Fe': [4], 'Gd': [1]}`；最终 CIF `{'Al': [4, 4], 'Fe': [4], 'Gd': [1]}`；匹配 `True`
- formation energy：预测 `-0.41` eV/atom；真实 prompt `-0.41` eV/atom；误差 `0.0`
- Fermi energy：预测 `7.03` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Al-Al | 2.793333 | 2.794722 | 0.001388 | True |
| Al-Fe | 2.61 | 2.604485 | 0.005515 | True |
| Al-Gd | 3.166667 | 3.122739 | 0.043928 | True |
| Fe-Fe | 2.53 | 2.530642 | 0.000642 | True |
| Fe-Gd | 3.36 | 3.368482 | 0.008482 | True |

最终简化 CIF：

```text
P1
5.06128300 6.75470723 6.75470723
81.5558 67.5646 67.5646
Gd 1 0.00000000 0.00000000 0.00000000
Al 1 0.00000000 0.66484400 0.33515600
Al 1 0.00000000 0.33515600 0.66484400
Al 1 0.66484400 0.33515600 0.33515600
Al 1 0.33515600 0.66484400 0.66484400
Al 1 0.50000000 0.77736900 0.22263100
Al 1 0.50000000 0.22263100 0.77736900
Al 1 0.72263100 0.77736900 0.77736900
Al 1 0.27736900 0.22263100 0.22263100
Fe 1 0.00000000 0.00000000 0.50000000
Fe 1 0.50000000 0.00000000 0.50000000
Fe 1 0.50000000 0.50000000 0.00000000
Fe 1 0.00000000 0.50000000 0.00000000
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_Gd(Al2Fe)4
_symmetry_space_group_name_H-M   I4/mmm
_cell_length_a   8.82981669
_cell_length_b   8.82981669
_cell_length_c   5.06128300
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   139
_chemical_formula_structural   Gd(Al2Fe)4
_chemical_formula_sum   'Gd2 Al16 Fe8'
_cell_volume   394.60628376
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-y, x, z'
  4  'y, -x, -z'
  5  '-x, -y, z'
  6  'x, y, -z'
  7  'y, -x, z'
  8  '-y, x, -z'
  9  'x, -y, -z'
  10  '-x, y, z'
  11  '-y, -x, -z'
  12  'y, x, z'
  13  '-x, y, -z'
  14  'x, -y, z'
  15  'y, x, -z'
  16  '-y, -x, z'
  17  'x+1/2, y+1/2, z+1/2'
  18  '-x+1/2, -y+1/2, -z+1/2'
  19  '-y+1/2, x+1/2, z+1/2'
  20  'y+1/2, -x+1/2, -z+1/2'
  21  '-x+1/2, -y+1/2, z+1/2'
  22  'x+1/2, y+1/2, -z+1/2'
  23  'y+1/2, -x+1/2, z+1/2'
  24  '-y+1/2, x+1/2, -z+1/2'
  25  'x+1/2, -y+1/2, -z+1/2'
  26  '-x+1/2, y+1/2, z+1/2'
  27  '-y+1/2, -x+1/2, -z+1/2'
  28  'y+1/2, x+1/2, z+1/2'
  29  '-x+1/2, y+1/2, -z+1/2'
  30  'x+1/2, -y+1/2, z+1/2'
  31  'y+1/2, x+1/2, -z+1/2'
  32  '-y+1/2, -x+1/2, z+1/2'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Gd  Gd0  2  0.00000000  0.00000000  0.00000000  1
  Al  Al1  8  0.00000000  0.22263100  0.50000000  1
  Al  Al2  8  0.00000000  0.33515600  0.00000000  1
  Fe  Fe3  8  0.25000000  0.25000000  0.25000000  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.12973198837612648, "sum_of_information": 70.18500571148442, "token_num": 541}` |
| gt | `{"comp": "Gd1 Al8 Fe4", "mp_id": "mp-1104719", "spg": 139}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-77.44548034667969` |
| energy_above_hull | `0.41617626564002386` |
| is_stable | `False` |
| is_novel | `True` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
5.02383155 6.64944751 6.64944760
81.8025 67.815 67.815
Gd 1 0.00000000 -0.00000000 -0.00000000
Al 1 -0.00000000 0.66414647 0.33585353
Al 1 -0.00000000 0.33585353 0.66414647
Al 1 0.66407748 0.33584343 0.33584343
Al 1 0.33592251 0.66415657 0.66415657
Al 1 0.50000000 0.77742701 0.22257298
Al 1 0.50000000 0.22257299 0.77742702
Al 1 0.72255795 0.77742640 0.77742640
Al 1 0.27744205 0.22257360 0.22257360
Fe 1 0.00000000 -0.00000000 0.50000000
Fe 1 0.50000000 -0.00000000 0.50000000
Fe 1 0.50000000 0.50000000 0.00000000
Fe 1 -0.00000000 0.50000000 -0.00000000
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 6: `mp-1212101` / row `14577`

- 总分：`99.632162`
- 空间群：预测 `Pnma` / `62`；最终 CIF `Pnma` / `62`；匹配 `True`
- 体积：预测 `250.4`；最终 CIF `250.472933`；相对误差 `0.029118%`
- site：预测 `{'Ho': [4], 'Ir': [4], 'Sb': [4]}`；最终 CIF `{'Ho': [4], 'Ir': [4], 'Sb': [4]}`；匹配 `True`
- formation energy：预测 `-0.889` eV/atom；真实 prompt `-0.8889` eV/atom；误差 `0.0001`
- Fermi energy：预测 `7.13` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Ho-Ir | 3.18 | 3.192861 | 0.012861 | True |
| Ho-Sb | 3.2 | 3.198588 | 0.001412 | True |
| Ir-Sb | 2.715 | 2.678465 | 0.036535 | True |

最终简化 CIF：

```text
P1
4.53519928 7.15136626 7.72281098
90 90 90
Ho 1 0.25000000 0.51317706 0.20397480
Ho 1 0.25000000 0.01317706 0.29602520
Ho 1 0.75000000 0.48682294 0.79602520
Ho 1 0.75000000 0.98682294 0.70397480
Sb 1 0.25000000 0.68973398 0.58796058
Sb 1 0.25000000 0.18973398 0.91203942
Sb 1 0.75000000 0.31026602 0.41203942
Sb 1 0.75000000 0.81026602 0.08796058
Ir 1 0.25000000 0.79730083 0.91422105
Ir 1 0.25000000 0.29730083 0.58577895
Ir 1 0.75000000 0.20269917 0.08577895
Ir 1 0.75000000 0.70269917 0.41422105
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_HoSbIr
_symmetry_space_group_name_H-M   Pnma
_cell_length_a   7.15136626
_cell_length_b   4.53519928
_cell_length_c   7.72281098
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   62
_chemical_formula_structural   HoSbIr
_chemical_formula_sum   'Ho4 Sb4 Ir4'
_cell_volume   250.47293315
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-x+1/2, -y, z+1/2'
  4  'x+1/2, y, -z+1/2'
  5  'x+1/2, -y+1/2, -z+1/2'
  6  '-x+1/2, y+1/2, z+1/2'
  7  '-x, y+1/2, -z'
  8  'x, -y+1/2, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ho  Ho0  4  0.01317706  0.25000000  0.70397480  1
  Sb  Sb1  4  0.18973398  0.25000000  0.08796058  1
  Ir  Ir2  4  0.20269917  0.75000000  0.91422105  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.2867417107493793, "sum_of_information": 141.07692168869463, "token_num": 492}` |
| gt | `{"comp": "Ho1 Sb1 Ir1", "mp_id": "mp-1212101", "spg": 62}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-80.46577453613281` |
| energy_above_hull | `0.04009385198893245` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
4.57437823 7.12865199 7.83466505
90 90 90
Ho 1 0.25000001 0.51309374 0.20343939
Ho 1 0.25000001 0.01309374 0.29656061
Ho 1 0.75000000 0.48690626 0.79656061
Ho 1 0.75000000 0.98690626 0.70343939
Sb 1 0.25000002 0.68949415 0.58802707
Sb 1 0.25000002 0.18949415 0.91197293
Sb 1 0.75000000 0.31050583 0.41197294
Sb 1 0.74999999 0.81050584 0.08802706
Ir 1 0.25000002 0.79704465 0.91435472
Ir 1 0.25000002 0.29704466 0.58564527
Ir 1 0.74999995 0.20295536 0.08564529
Ir 1 0.74999996 0.70295536 0.41435473
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 7: `mp-19417` / row `13026`

- 总分：`99.49092`
- 空间群：预测 `R-3` / `148`；最终 CIF `R-3` / `148`；匹配 `True`
- 体积：预测 `105.97`；最终 CIF `106.060066`；相对误差 `0.08492%`
- site：预测 `{'Fe': [2], 'O': [6], 'Ti': [2]}`；最终 CIF `{'Fe': [2], 'O': [6], 'Ti': [2]}`；匹配 `True`
- formation energy：预测 `-2.718` eV/atom；真实 prompt `-2.7177` eV/atom；误差 `0.0003`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Fe-O | 2.065 | 2.060637 | 0.004363 | True |
| O-Ti | 2.005 | 2.043053 | 0.038053 | True |

最终简化 CIF：

```text
P1
5.51233272 5.51233272 5.51233318
55.4988 55.4988 55.4988
Ti 1 0.63975000 0.63975000 0.63975000
Ti 1 0.36025000 0.36025000 0.36025000
Fe 1 0.84232500 0.84232500 0.84232500
Fe 1 0.15767500 0.15767500 0.15767500
O 1 0.98371600 0.23988200 0.54585100
O 1 0.54585100 0.98371600 0.23988200
O 1 0.23988200 0.54585100 0.98371600
O 1 0.01628400 0.76011800 0.45414900
O 1 0.45414900 0.01628400 0.76011800
O 1 0.76011800 0.45414900 0.01628400
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_TiFeO3
_symmetry_space_group_name_H-M   R-3
_cell_length_a   5.13314225
_cell_length_b   5.13314225
_cell_length_c   13.94363203
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   120.00000000
_symmetry_Int_Tables_number   148
_chemical_formula_structural   TiFeO3
_chemical_formula_sum   'Ti6 Fe6 O18'
_cell_volume   318.18019498
_cell_formula_units_Z   6
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-y, x-y, z'
  4  'y, -x+y, -z'
  5  '-x+y, -x, z'
  6  'x-y, x, -z'
  7  'x+2/3, y+1/3, z+1/3'
  8  '-x+2/3, -y+1/3, -z+1/3'
  9  '-y+2/3, x-y+1/3, z+1/3'
  10  'y+2/3, -x+y+1/3, -z+1/3'
  11  '-x+y+2/3, -x+1/3, z+1/3'
  12  'x-y+2/3, x+1/3, -z+1/3'
  13  'x+1/3, y+2/3, z+2/3'
  14  '-x+1/3, -y+2/3, -z+2/3'
  15  '-y+1/3, x-y+2/3, z+2/3'
  16  'y+1/3, -x+y+2/3, -z+2/3'
  17  '-x+y+1/3, -x+2/3, z+2/3'
  18  'x-y+1/3, x+2/3, -z+2/3'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ti  Ti0  6  0.00000000  0.00000000  0.36025000  1
  Fe  Fe1  6  0.00000000  0.00000000  0.15767500  1
  O  O2  18  0.01660100  0.28936800  0.74351700  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.21410821022668594, "sum_of_information": 91.85242218724827, "token_num": 429}` |
| gt | `{"comp": "Ti1 Fe1 O3", "mp_id": "mp-19417", "spg": 148}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-80.87713623046875` |
| energy_above_hull | `0.02047841795312344` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
5.64727671 5.64727595 5.64727714
54.0657 54.0657 54.0656
Ti 1 0.64425499 0.64425502 0.64425499
Ti 1 0.35574500 0.35574500 0.35574499
Fe 1 0.85052777 0.85052773 0.85052776
Fe 1 0.14947226 0.14947226 0.14947225
O 1 0.93160039 0.27948574 0.55865198
O 1 0.55865198 0.93160037 0.27948575
O 1 0.27948574 0.55865200 0.93160036
O 1 0.06839964 0.72051425 0.44134800
O 1 0.44134799 0.06839964 0.72051425
O 1 0.72051424 0.44134798 0.06839968
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 8: `mp-1227742` / row `70`

- 总分：`99.406812`
- 空间群：预测 `P4/mmm` / `123`；最终 CIF `P4/mmm` / `123`；匹配 `True`
- 体积：预测 `141.2`；最终 CIF `140.939213`；相对误差 `0.185035%`
- site：预测 `{'Ba': [1], 'O': [1, 1, 4], 'Sn': [2], 'Sr': [1]}`；最终 CIF `{'Ba': [1], 'O': [1, 1, 4], 'Sn': [2], 'Sr': [1]}`；匹配 `True`
- formation energy：预测 `-2.588` eV/atom；真实 prompt `-2.5884` eV/atom；误差 `0.0004`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Ba-O | 2.926667 | 2.90461 | 0.022057 | True |
| O-Sn | 2.05 | 2.065569 | 0.015569 | True |
| O-Sr | 2.913333 | 2.936931 | 0.023597 | True |

最终简化 CIF：

```text
P1
4.13673900 4.13673900 8.23598600
90 90 90
Ba 1 0.00000000 0.00000000 0.00000000
Sr 1 0.00000000 0.00000000 0.50000000
Sn 1 0.50000000 0.50000000 0.25127200
Sn 1 0.50000000 0.50000000 0.74872800
O 1 0.50000000 0.50000000 0.00000000
O 1 0.50000000 0.50000000 0.50000000
O 1 0.50000000 0.00000000 0.24582800
O 1 0.50000000 0.00000000 0.75417200
O 1 0.00000000 0.50000000 0.24582800
O 1 0.00000000 0.50000000 0.75417200
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_BaSr(SnO3)2
_symmetry_space_group_name_H-M   P4/mmm
_cell_length_a   4.13673900
_cell_length_b   4.13673900
_cell_length_c   8.23598600
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   123
_chemical_formula_structural   BaSr(SnO3)2
_chemical_formula_sum   'Ba1 Sr1 Sn2 O6'
_cell_volume   140.93921271
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-y, x, z'
  4  'y, -x, -z'
  5  '-x, -y, z'
  6  'x, y, -z'
  7  'y, -x, z'
  8  '-y, x, -z'
  9  'x, -y, -z'
  10  '-x, y, z'
  11  '-y, -x, -z'
  12  'y, x, z'
  13  '-x, y, -z'
  14  'x, -y, z'
  15  'y, x, -z'
  16  '-y, -x, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ba  Ba0  1  0.00000000  0.00000000  0.00000000  1
  Sr  Sr1  1  0.00000000  0.00000000  0.50000000  1
  Sn  Sn2  2  0.50000000  0.50000000  0.25127200  1
  O  O3  4  0.00000000  0.50000000  0.24582800  1
  O  O4  1  0.50000000  0.50000000  0.00000000  1
  O  O5  1  0.50000000  0.50000000  0.50000000  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.10924343151591105, "sum_of_information": 45.22678064758718, "token_num": 414}` |
| gt | `{"comp": "Ba1 Sr1 Sn2 O6", "mp_id": "mp-1227742", "spg": 123}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-63.014434814453125` |
| energy_above_hull | `0.030601250304687078` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
4.15385153 4.15385163 8.30584622
90 90 90
Ba 1 -0.00000000 -0.00000000 -0.00000000
Sr 1 0.00000000 0.00000000 0.50000000
Sn 1 0.50000000 0.50000000 0.24984200
Sn 1 0.50000000 0.50000000 0.75015800
O 1 0.50000000 0.50000000 0.00000000
O 1 0.50000000 0.50000000 0.50000000
O 1 0.50000000 0.00000000 0.24844150
O 1 0.50000000 -0.00000000 0.75155850
O 1 -0.00000000 0.50000000 0.24844150
O 1 -0.00000000 0.50000000 0.75155850
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 9: `mp-753496` / row `7432`

- 总分：`99.339864`
- 空间群：预测 `Imma` / `74`；最终 CIF `Imma` / `74`；匹配 `True`
- 体积：预测 `152.86`；最终 CIF `152.559331`；相对误差 `0.197083%`
- site：预测 `{'Co': [2], 'Li': [2], 'O': [4, 4], 'Sn': [2]}`；最终 CIF `{'Co': [2], 'Li': [2], 'O': [4, 4], 'Sn': [2]}`；匹配 `True`
- formation energy：预测 `-1.843` eV/atom；真实 prompt `-1.8429` eV/atom；误差 `0.0001`
- Fermi energy：预测 `None` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Co-O | 2.006667 | 2.025989 | 0.019322 | True |
| Li-O | 1.985 | 1.948834 | 0.036166 | True |
| O-Sn | 2.083333 | 2.097303 | 0.01397 | True |

最终简化 CIF：

```text
P1
6.00141210 6.00141210 6.00141210
120.766 119.343 90.0059
Li 1 0.87843500 0.12843500 0.75000000
Li 1 0.12156500 0.87156500 0.25000000
Co 1 0.50000000 0.50000000 0.50000000
Co 1 0.50000000 0.00000000 0.00000000
Sn 1 0.50000000 0.50000000 0.00000000
Sn 1 0.00000000 0.50000000 0.50000000
O 1 0.25896000 0.24782300 0.01113700
O 1 0.73059000 0.24782300 0.48886300
O 1 0.72847500 0.24450800 0.01396700
O 1 0.72847500 0.71450800 0.48603300
O 1 0.74104000 0.75217700 0.98886300
O 1 0.26941000 0.75217700 0.51113700
O 1 0.27152500 0.75549200 0.98603300
O 1 0.27152500 0.28549200 0.51396700
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_LiCoSnO4
_symmetry_space_group_name_H-M   Imma
_cell_length_a   5.93179367
_cell_length_b   6.06091047
_cell_length_c   8.48684139
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   74
_chemical_formula_structural   LiCoSnO4
_chemical_formula_sum   'Li4 Co4 Sn4 O16'
_cell_volume   305.11951851
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-x, -y+1/2, z'
  4  'x, y+1/2, -z'
  5  'x, -y, -z'
  6  '-x, y, z'
  7  '-x, y+1/2, -z'
  8  'x, -y+1/2, z'
  9  'x+1/2, y+1/2, z+1/2'
  10  '-x+1/2, -y+1/2, -z+1/2'
  11  '-x+1/2, -y, z+1/2'
  12  'x+1/2, y, -z+1/2'
  13  'x+1/2, -y+1/2, -z+1/2'
  14  '-x+1/2, y+1/2, z+1/2'
  15  '-x+1/2, y, -z+1/2'
  16  'x+1/2, -y, z+1/2'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Li  Li0  4  0.00000000  0.25000000  0.12843500  1
  Co  Co1  4  0.25000000  0.25000000  0.75000000  1
  Sn  Sn2  4  0.00000000  0.00000000  0.50000000  1
  O  O3  8  0.00000000  0.01113700  0.75217700  1
  O  O4  8  0.23500000  0.25000000  0.52049200  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.19892365464146347, "sum_of_information": 114.77894872812442, "token_num": 577}` |
| gt | `{"comp": "Li1 Co1 Sn1 O4", "mp_id": "mp-753496", "spg": 74}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-82.6895980834961` |
| energy_above_hull | `2.4179394079645613` |
| is_stable | `False` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
5.99016711 5.99002931 5.97773721
121.789 117.647 90.5916
Li 1 0.87705793 0.12711274 0.74994063
Li 1 0.12294206 0.87288726 0.25005937
Co 1 0.50000000 0.50000000 0.50000000
Co 1 0.50000000 0.00000000 0.00000001
Sn 1 0.49999999 0.50000000 -0.00000001
Sn 1 -0.00000001 0.50000000 0.49999999
O 1 0.27258333 0.25061029 0.02092432
O 1 0.72536567 0.25135161 0.47822684
O 1 0.72879985 0.24271008 0.01287674
O 1 0.72943284 0.71603899 0.48759956
O 1 0.72741668 0.74938971 0.97907568
O 1 0.27463433 0.74864840 0.52177316
O 1 0.27120015 0.75728993 0.98712327
O 1 0.27056717 0.28396101 0.51240045
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。

### 样本 10: `mp-1919` / row `14257`

- 总分：`99.224891`
- 空间群：预测 `P6_3/mmc` / `194`；最终 CIF `P6_3/mmc` / `194`；匹配 `True`
- 体积：预测 `179.1`；最终 CIF `178.191829`；相对误差 `0.509659%`
- site：预测 `{'Cr': [2, 6], 'Zr': [4]}`；最终 CIF `{'Cr': [2, 6], 'Zr': [4]}`；匹配 `True`
- formation energy：预测 `0.188` eV/atom；真实 prompt `0.1884` eV/atom；误差 `0.0004`
- Fermi energy：预测 `5.49` eV；真实值 `None`；说明 `The final CIF and conditional+thinking parquet do not contain a real Fermi energy label.`

键长逐 pair 对比：

| pair | 预测均值 Å | 最终 CIF 均值 Å | 绝对误差 Å | 是否 <=0.05 Å |
| --- | --- | --- | --- | --- |
| Cr-Cr | 2.506 | 2.506911 | 0.000911 | True |
| Cr-Zr | 2.965 | 2.939366 | 0.025634 | True |

最终简化 CIF：

```text
P1
5.03616671 5.03616671 8.11254200
90 90 120
Zr 1 0.33333300 0.66666700 0.06385800
Zr 1 0.66666700 0.33333300 0.56385800
Zr 1 0.66666700 0.33333300 0.93614200
Zr 1 0.33333300 0.66666700 0.43614200
Cr 1 0.00000000 0.00000000 0.00000000
Cr 1 0.00000000 0.00000000 0.50000000
Cr 1 0.83326300 0.16673700 0.25000000
Cr 1 0.83326300 0.66652500 0.25000000
Cr 1 0.33347500 0.16673700 0.25000000
Cr 1 0.16673700 0.83326300 0.75000000
Cr 1 0.16673700 0.33347500 0.75000000
Cr 1 0.66652500 0.83326300 0.75000000
```

Symmetrized real CIF：

- 生成器：`pymatgen`
- 说明：Generated with pymatgen SpacegroupAnalyzer.get_refined_structure() and CifWriter.

```cif
# generated using pymatgen
data_ZrCr2
_symmetry_space_group_name_H-M   P6_3/mmc
_cell_length_a   5.03616671
_cell_length_b   5.03616671
_cell_length_c   8.11254200
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   120.00000000
_symmetry_Int_Tables_number   194
_chemical_formula_structural   ZrCr2
_chemical_formula_sum   'Zr4 Cr8'
_cell_volume   178.19182910
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  'x-y, x, z+1/2'
  4  '-x+y, -x, -z+1/2'
  5  '-y, x-y, z'
  6  'y, -x+y, -z'
  7  '-x, -y, z+1/2'
  8  'x, y, -z+1/2'
  9  '-x+y, -x, z'
  10  'x-y, x, -z'
  11  'y, -x+y, z+1/2'
  12  '-y, x-y, -z+1/2'
  13  '-y, -x, -z+1/2'
  14  'y, x, z+1/2'
  15  '-x, -x+y, -z'
  16  'x, x-y, z'
  17  '-x+y, y, -z+1/2'
  18  'x-y, -y, z+1/2'
  19  'y, x, -z'
  20  '-y, -x, z'
  21  'x, x-y, -z+1/2'
  22  '-x, -x+y, z+1/2'
  23  'x-y, -y, -z'
  24  '-x+y, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Zr  Zr0  4  0.33333333  0.66666667  0.06385800  1
  Cr  Cr1  6  0.16673700  0.33347400  0.75000000  1
  Cr  Cr2  2  0.00000000  0.00000000  0.00000000  1
```

Parquet metrics：

| metric | value |
| --- | --- |
| extra_args | `{"average_of_information": 0.10223492293461917, "sum_of_information": 50.401817006767246, "token_num": 493}` |
| gt | `{"comp": "Zr1 Cr2", "mp_id": "mp-1919", "spg": 194}` |
| smact_validity | `True` |
| structure_validity | `True` |
| composition_consistency | `True` |
| spacegroup_consistency | `True` |
| energies | `-111.55509948730469` |
| energy_above_hull | `0.02077195939127563` |
| is_stable | `True` |
| is_novel | `False` |
| is_unique | `False` |
| stable_unique_novel | `False` |

`relaxed_structures`：

```text
P1
5.07337861 5.07337842 8.17825558
90 90 120
Zr 1 0.33333321 0.66666678 0.06206336
Zr 1 0.66666678 0.33333322 0.56206336
Zr 1 0.66666679 0.33333322 0.93793664
Zr 1 0.33333321 0.66666677 0.43793664
Cr 1 -0.00000000 0.00000000 0.00000000
Cr 1 0.00000000 -0.00000000 0.50000000
Cr 1 0.83243682 0.16756317 0.25000000
Cr 1 0.83243674 0.66487283 0.25000000
Cr 1 0.33512721 0.16756329 0.25000000
Cr 1 0.16756318 0.83243682 0.75000000
Cr 1 0.16756328 0.33512719 0.75000000
Cr 1 0.66487278 0.83243671 0.75000000
```

完整 response 已保存在 `best_thinking_consistency_cases.json`。
