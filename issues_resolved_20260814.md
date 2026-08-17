# Kết quả xử lý checklist trước khi viết

> **Nguồn checklist:** `issues_to_resolve_before_writing.md`
> **Analysis:** `outputs/analysis_v2/framing_v2_native_visdrone_20260814`
> **Artefact mới:** `outputs/issue_resolution/partab_20260814/`
> **Script:** [scripts/audit_analysis.py](scripts/audit_analysis.py) ·
> [scripts/bootstrap_ap_ci.py](scripts/bootstrap_ap_ci.py)
> **Ngày:** 14/08/2026

Không chạy lại inference, không build engine, không đổi protocol đã freeze.
Mọi số dưới đây trace về CSV/JSON trong `outputs/issue_resolution/partab_20260814/`.

---

## 0. Điều quan trọng nhất: checklist đánh giá thấp những gì đã có

Bốn mục checklist ghi là "chưa chạy / vẫn thiếu" thì **thực tế đã có sẵn trong
repo**, chỉ chưa được tổng hợp lên tầng paper:

| Mục | Checklist nói | Thực tế |
|---|---|---|
| B1 pipeline validation | "chưa chạy", cần GPU nửa ngày | `outputs/*/trt_fp16/` + `pipeline_controls.csv` đã có, **pass** |
| B2 GT instance count | "vẫn thiếu" | `validation_statistics.csv` đã có đủ |
| B3 false positive accounting | "vẫn thiếu", nửa ngày | `false_positive_counts.csv` đã có cho **cả 3 dataset** × 5 seed |
| A3/A4 phân bố Δs | "cần chạy" | `continuous_shift_summary.csv` đã có cho nhóm survivor |

Phần thật sự phải tính mới trong lượt đầu chỉ có hai thứ: **histogram score theo
scale** (A1) và **Δs của nhóm flip** (A4). Cả hai đều offline, và đều đã chạy xong.

Sau đó bổ sung ba mục nữa — C1 (tính diễn giải được của Table 3), C2 (tách nguồn
variance), C3 (bootstrap CI) — trong đó **bootstrap CI của QFR cũng đã có sẵn** từ
trước, chỉ CI của AP theo scale là phải chạy mới.

---

## A1 — Stratification coverage: chẩn đoán ĐÚNG, và mạnh hơn dự kiến

**Verdict: pass.** Không có bug tính bin — `frozen_edges_reproduced = true` cho cả
3 dataset, nghĩa là edge trong `confidence_bin_edges.json` tái tạo chính xác từ
phân bố pooled. Chẩn đoán trong checklist được xác nhận.

Bằng chứng định lượng — tỉ lệ mỗi scale nằm trong **2 decile cao nhất**:

| Dataset | Small | Medium | Large | Tỉ lệ L/S |
|---|---:|---:|---:|---:|
| COCO | 0.75% | 9.85% | **40.67%** | 54× |
| TT100K | 2.36% | 23.28% | **50.19%** | 21× |
| VisDrone | 2.06% | 32.01% | **78.18%** | 38× |

Phân bố score FP32-matched gần như rời nhau. VisDrone: large có q1 = 0.871 trong
khi small có q3 = 0.606 — **hai khoảng tứ phân vị không giao nhau**.

| Dataset | Scale | n | q1 | median | q3 |
|---|---|---:|---:|---:|---:|
| VisDrone | Small | 8,307 | 0.296 | 0.423 | 0.606 |
| VisDrone | Medium | 7,950 | 0.525 | 0.765 | 0.873 |
| VisDrone | Large | 921 | **0.871** | 0.923 | 0.937 |

### Điểm mới, quyết định lựa chọn Option

Em test luôn Option B (đổi sang bin thô hơn) — **nó không cứu được**. Áp cùng tỉ lệ
coverage 8/10 sang các mức bin thô hơn:

| Dataset | 10 bin (freeze) | 5 bin | 4 bin | 3 bin |
|---|---|---|---|---|
| COCO | 8/8 ✅ | 4/4 ✅ | 4/4 ✅ | 3/3 ✅ |
| TT100K | 1/8 ❌ | 3/4 ❌ | 3/4 ❌ | 3/3 ✅ |
| VisDrone | 2/8 ❌ | 3/4 ❌ | 2/4 ❌ | **2/3 ❌** |

TT100K chỉ cứu được ở mức **tertile** — thô đến mức khó gọi là difficulty control.
VisDrone **fail ở mọi mức bin**, kể cả 3 bin.

Option C cũng không giúp: khoảng common support `[0.20, 0.92]` trông rộng, nhưng
vấn đề là **mật độ chứ không phải biên** — chỉ 46.7% large object của VisDrone nằm
trong khoảng đó.

> **Kết luận: chọn Option A, và giờ nó có bằng chứng chứ không phải là lựa chọn cho
> tiện.** Difficulty control không khả thi trên TT100K/VisDrone, và việc làm thô bin
> không sửa được — vì nguyên nhân là phân bố confidence theo scale không chồng lấp,
> không phải vì bin quá mịn. Đây chính là contribution mà PART F điểm 2 nói tới, và
> bây giờ nó được chống lưng bằng 3 bảng.

Artefact: `a1_score_histogram_by_scale.csv`, `a1_score_distribution_by_scale.csv`,
`a1_binning_variants.csv`, `a1_common_support.csv`, `a1_diagnosis.json`.

---

## A2 — KHÔNG phải blocker. FP32 baseline chưa từng đổi

**Verdict: pass.** Tiền đề của A2 sai.

1. **`predictions.json` của FP32 giống nhau bit-for-bit** giữa `fp32/` và
   `fp32_repro_v2/` trên cả 3 dataset (sha256 khớp). FP32 inference deterministic
   đúng như phải vậy.
2. **Native evaluator khớp pycocotools**: COCO lệch `0.0`, TT100K lệch `−1.1e−16`.
   VisDrone lệch `−1.64e−3`, đúng bằng hiệu ứng ignore region đã ghi nhận.
3. **Chênh lệch 0.0521 vs 0.0558 không tồn tại** trong chain hiện tại:
   `paper_ap_table.delta_ap_mean` = `ap_decomposition.total_drop_mean` = 0.052109,
   khớp tuyệt đối (diff = 0.0) cho cả 3 dataset. Grep toàn bộ git history không tìm
   thấy nguồn nào sinh ra 0.0558 → đó là số tồn từ một lần chạy cũ đã bị thay thế.
4. **Khác biệt duy nhất giữa hai analysis là VisDrone**, và chỉ ở các metric mà
   ignore region ảnh hưởng:

| Dataset | Metric | r1 | native | Δ |
|---|---|---:|---:|---:|
| visdrone | AP | 0.1853 | 0.1872 | +0.0020 |
| visdrone | AP_S | 0.0979 | 0.0999 | +0.0020 |
| visdrone | AP_M | 0.2846 | 0.2867 | +0.0021 |
| visdrone | AP_L | 0.5293 | 0.5293 | **0.0000** |

AP_L không đổi — hợp lý, ignore region của VisDrone là object nhỏ. COCO và TT100K
**không đổi một chữ số nào**.

> Bảng ΔAP trong mục A2 của checklist (+0.0052 / +0.0047 / −0.0004) không khớp với
> bất kỳ artefact nào trong repo. Nó không phải "FP32 shift" mà là số từ nguồn cũ.
> Baseline chốt: **pycocotools COCOeval trong analysis_v2 chạy trên
> `fp32/predictions.json`**. Mọi bảng/hình đã dùng đúng baseline này.

Artefact: `a2_fp32_baseline_provenance.json`, `a2_cross_analysis_comparison.csv`,
`a2_total_drop_reconciliation.csv`.

---

## A3 — Giả thuyết score inflation bị BÁC BỎ

**Verdict: pass, nhưng kết quả ngược giả thuyết.**

Δs = s_INT8 − s_FP32 cho nhóm survivor (transition 11), gộp 5 seed:

| Dataset | Scale | n | median Δs | ±std | q1 | q3 | % âm |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | Small | 2,739 | −0.072 | 0.005 | −0.146 | 0.000 | 74.9% |
| COCO | Medium | 6,327 | −0.094 | 0.020 | −0.198 | −0.010 | 77.2% |
| COCO | Large | 5,338 | **−0.320** | 0.042 | −0.417 | −0.225 | **97.4%** |
| TT100K | Small | 1,187 | −0.051 | 0.015 | −0.134 | 0.016 | 68.9% |
| TT100K | Medium | 2,564 | −0.026 | 0.008 | −0.091 | 0.015 | 66.6% |
| TT100K | Large | 490 | −0.023 | 0.005 | −0.088 | 0.002 | 72.7% |
| VisDrone | Small | 3,628 | −0.258 | 0.193 | −0.346 | −0.131 | 81.9% |
| VisDrone | Medium | 5,691 | −0.221 | 0.143 | −0.335 | −0.022 | 80.5% |
| VisDrone | Large | 892 | **+0.002** | 0.005 | −0.021 | 0.017 | 47.3% |

Theo bảng pass/fail của checklist: Δs của large **lệch âm rõ** trên COCO (−0.320,
97.4% âm) và **≈ 0** trên VisDrone (+0.002, 47.3% âm). Không có dataset nào lệch
dương. → **QRR_L cao không đến từ score inflation.**

Δs của IoU rất nhỏ ở mọi cell (median −0.008 … −0.022) → cũng không phải box shift
làm IoU vượt τ.

> **Cơ chế thật, và nó gọn hơn giả thuyết ban đầu:** quantization làm **giảm** score,
> mức giảm **tăng theo scale** trên COCO (S −0.07 < M −0.09 < L −0.32). Large object
> tụt score mạnh nhất — nhưng chúng khởi điểm rất cao (median FP32 0.855 trên COCO,
> 0.923 trên VisDrone) nên vẫn còn dư địa trên ngưỡng t = 0.2. Small object tụt ít
> hơn nhiều nhưng khởi điểm sát ngưỡng (median 0.42) nên vượt ngưỡng và thành flip.
>
> Một cơ chế duy nhất giải thích cả ba nghịch lý: ΔAP_L lớn nhất (score collapse phá
> ranking, khớp 28–41% ranking share), QFR_L nhỏ (large còn dư địa), và QFR_S lớn
> (small không có dư địa).

Artefact: `a3_survivor_score_shift.csv`.

---

## A4 — Failure mode: không tautological, nhưng phải phát biểu lại

**Verdict: pass với điều kiện.**

### A4a. Δs của nhóm flip (tính mới)

Với mỗi flip 10 được xếp mode `confidence`, em lấy score của evidence box còn sống
(`evidence_prediction_id` → `predictions.json`) và so với `fp32_score`:

| Dataset | Scale | n/seed | median s_FP32 | median s_INT8 | median Δs | grazed (≥0.10) | collapsed (<0.05) |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | Small | 851.6 | 0.272 | 0.136 | −0.153 | 70.4% | 9.7% |
| COCO | Medium | 1240.0 | 0.336 | 0.116 | −0.233 | 57.6% | 17.9% |
| COCO | Large | 1717.0 | 0.516 | 0.097 | **−0.415** | 48.1% | 26.3% |
| TT100K | Small | 204.2 | 0.296 | 0.134 | −0.184 | 66.8% | 13.4% |
| TT100K | Medium | 206.6 | 0.330 | 0.129 | −0.220 | 67.3% | 12.3% |
| TT100K | Large | **27.2** | 0.362 | 0.089 | −0.276 | 44.1% | 27.2% |
| VisDrone | Small | 4441.4 | 0.358 | 0.070 | −0.291 | 35.5% | 38.2% |
| VisDrone | Medium | 2179.2 | 0.510 | 0.082 | −0.435 | 41.1% | 30.5% |
| VisDrone | Large | **26.2** | 0.610 | 0.119 | −0.495 | 55.0% | 20.6% |

Đọc theo bảng pass/fail của checklist thì kết quả **nằm giữa hai ô**:

- Score **không** chỉ lướt qua ngưỡng: median Δs từ −0.15 đến −0.50, tức tụt
  45–81% giá trị tương đối. Đây là sụt thật.
- Nhưng score **cũng không** sụt về 0: median điểm hạ cánh là 0.07–0.14, và
  35–70% flip nằm trong dải `[0.10, 0.20)` — tức trong vòng nửa ngưỡng.

> **Phát biểu đúng:** cơ chế là score compression phụ thuộc scale, độ lớn tăng theo
> scale. Việc nó *trở thành* flip hay không lại do khoảng cách từ score FP32 tới
> operating point quyết định. Nên Figure 3 phải nói cả hai, và QFR phải được ghi rõ
> là đại lượng tại một operating point cố định — đúng như dòng cuối trong
> Limitations đã dự trù.

Về lo ngại tautological: mode `confidence` chiếm 95–99.9% **một phần là do định
nghĩa** (nó được kiểm tra trước, và hầu như luôn còn một box cùng class ở IoU ≥ τ).
Phần mang thông tin không nằm ở tỉ lệ 99% mà ở **độ lớn Δs** trong bảng trên.

> **Đề xuất cho Figure 3:** thay stacked bar 4 mode (gần như phẳng 99% ở mọi cell,
> không mang thông tin) bằng phân bố Δs theo scale, gộp survivor (A3) và flip (A4a),
> vẽ kèm đường t = 0.2. Đây là hình mang cơ chế thật của paper — đúng như PART F
> điểm 3 dự đoán. Bảng 4 mode xuống appendix.

### A4b. Threshold sweep đầy đủ 5 × 2

Tổng hợp xong toàn bộ 30 cell (5 score × 2 IoU × 3 dataset):

| Dataset | IoU | ordering qua 5 mức t | small_highest | small_lowest | ổn định |
|---|---|---|---:|---:|---|
| COCO | 0.50 | `L>S>M` ↔ `S>L>M` | 2/5 | 0/5 | ❌ |
| COCO | 0.75 | `L>S>M` ↔ `S>L>M` | 3/5 | 0/5 | ❌ |
| TT100K | 0.50 | `S>M>L` | **5/5** | 0/5 | ✅ |
| TT100K | 0.75 | `S>M>L` | **5/5** | 0/5 | ✅ |
| VisDrone | 0.50 | `S>M>L` | **5/5** | 0/5 | ✅ |
| VisDrone | 0.75 | `S>M>L` | **5/5** | 0/5 | ✅ |

Hai kết quả, cả hai đều dùng được:

1. **`small_lowest` = 0/30 cell.** Trên toàn bộ sweep đã freeze, raw QFR **chưa bao
   giờ** nói small là scale bền nhất. Kết luận ở t = 0.2 không phải artefact của
   ngưỡng — mạnh hơn hẳn một dòng ở t = 0.2.
2. **TT100K và VisDrone ổn định hoàn hảo** (`S>M>L` ở cả 10/10 cell), còn **COCO đảo
   thứ tự S↔L theo t**. Bản thân việc đó là finding về threshold-dependence, đúng ô
   thứ ba trong bảng pass/fail. Ghi vào Results + bảng appendix.

Artefact: `a4_flip_score_shift.csv`, `a4_flip_score_shift_per_seed.csv`,
`a4_threshold_sweep_ordering.csv`, `a4_threshold_sweep_stability.csv`.

---

## B1 — Pipeline validation: PASS, đã có sẵn, không cần GPU

**Verdict: pass.** 12/12 cell nằm trong dải ±1% relative AP.

| Dataset | AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|
| COCO | −0.49% | −0.33% | −0.19% | −0.51% |
| TT100K | +0.33% | **+0.88%** | +0.32% | −0.09% |
| VisDrone | +0.13% | −0.49% | +0.68% | +0.26% |

Lệch lớn nhất 0.88% (TT100K AP_S) < 1% → pass. Và vì control là **PyTorch FP32 →
TensorRT FP16**, nó đồng thời phủ luôn yêu cầu "backend thứ hai": đổi cả runtime lẫn
precision mà AP chênh < 1 AP point. Hai gạch đầu dòng B1 đóng bằng một control.

Engine audit (`engine_audit.csv`): cả 15 engine INT8 đều có Q/DQ graph
(131–142 quantize layer, 114–125 weight-quantize layer); 3 engine FP16 control có 0
— đúng như kỳ vọng.

**Còn lại một hạn chế thật:** scheme per-channel **không xác nhận được** từ engine đã
build, vì engine build với profiling verbosity chỉ có layer name. Đây là mục duy nhất
của B1 chưa đóng, và nó phải vào Limitations chứ không thể phát biểu như đã kiểm tra.

Artefact: `b1_fp16_control.csv`, `b1_engine_audit_summary.csv`.

---

## B2 — GT instance count: xong

| Dataset | #Images | #Objects | #S | #M | #L | %S | %M | %L | obj/img | Median area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO | 5,000 | 36,335 | 15,264 | 12,392 | 8,679 | 42.0 | 34.1 | 23.9 | 7.3 | 1,641.2 |
| TT100K | 3,016 | 7,575 | 3,318 | 3,679 | 578 | 43.8 | 48.6 | **7.6** | 2.5 | 1,248.1 |
| VisDrone | 548 | 38,759 | 26,575 | 11,116 | 1,068 | 68.6 | 28.7 | **2.8** | 70.7 | 520.0 |

Số này giải thích trực tiếp A1: large chỉ chiếm 7.6% (TT100K) và 2.8% (VisDrone)
tổng GT, nên khi decile edge cắt trên phân bố pooled thì large không đủ mật độ.
Vào caption Table 1.

Artefact: `b2_gt_instance_counts.csv`.

---

## B3 — False positive: INT8 sinh ÍT FP hơn FP32

**Verdict: pass, và nó loại bỏ một cách diễn giải sai.**

Trung bình 5 seed, gán theo diện tích box dự đoán:

| Dataset | Scale | FP32 FP | INT8 FP | ΔFP | INT8-only | % chỉ có ở INT8 | FP32/img | INT8/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | Small | 1,467 | 922 | −545 | 266 | 28.8% | 0.29 | 0.18 |
| COCO | Medium | 3,123 | 2,641 | −482 | 961 | 36.2% | 0.62 | 0.53 |
| COCO | Large | 2,879 | 921 | **−1,958** | 266 | 28.9% | 0.58 | 0.18 |
| TT100K | Small | 1,034 | 884 | −150 | 302 | 34.0% | 0.34 | 0.29 |
| TT100K | Medium | 1,170 | 1,089 | −81 | 338 | 31.0% | 0.39 | 0.36 |
| TT100K | Large | 103 | 104 | **+1** | 37 | 35.6% | 0.03 | 0.03 |
| VisDrone | Small | 6,087 | 2,475 | −3,612 | 994 | 40.7% | 11.11 | 4.52 |
| VisDrone | Medium | 3,257 | 1,684 | −1,573 | 595 | 33.7% | 5.94 | 3.07 |
| VisDrone | Large | 175 | 216 | **+41** | 88 | 40.6% | 0.32 | 0.39 |

> **ΔAP_L trên COCO KHÔNG do FP tăng** — FP large giảm 2,879 → 921. Điều kiện "nếu
> ΔAP_L chủ yếu do FP tăng thì phải viết lại diễn giải" **không xảy ra**. Phần diễn
> giải giữ nguyên.
>
> Và nó khớp hoàn hảo với A3: score sụt trên toàn bộ phân bố nên FP cũng tụt xuống
> dưới ngưỡng. INT8 mất cả detection thật lẫn FP. Đó là lý do AP giảm dù FP ít hơn —
> mất recall cộng với ranking bị phá, chứ không phải nhiễu thêm.

Artefact: `b3_false_positive_accounting.csv`.

---

## C1 — Table 3: nối với A1, nhưng nguyên nhân chính là calibration bimodal

**Verdict: phải vá. Và chẩn đoán "regression extrapolate ngoài vùng dữ liệu" không
đúng.** Em đo VIF trên chính design matrix mà `analysis_statistics._design` dùng:

| Dataset | Term | VIF | flip events | overlap với small | Kết luận |
|---|---|---:|---:|---:|---|
| COCO | scale_medium | 1.96 | 1,236 | 0.744 | seed mean dùng được |
| COCO | scale_large | 2.35 | 1,966 | 0.435 | seed mean dùng được |
| TT100K | scale_medium | 1.68 | 231 | 0.689 | seed mean dùng được |
| TT100K | scale_large | 1.67 | **26** | 0.442 | **per-seed only** |
| VisDrone | scale_medium | 1.45 | 4,196 | 0.583 | **per-seed only** |
| VisDrone | scale_large | 1.31 | **35** | **0.170** | **per-seed only** |

Collinearity nhẹ ở mọi nơi (VIF ≤ 2.35). Nếu paper viết "extrapolate ngoài vùng dữ
liệu", reviewer chạy VIF ra 1.31 thì đó là điểm trừ mới. Nguyên nhân thật:

**(a) Calibration bimodal — nguyên nhân chính.** CI trong từng seed đều chặt nhưng
ngược dấu nhau giữa các seed:

```
visdrone / scale_large
  s0: β = −4.257  CI = [−4.955, −3.733]  NEG
  s1: β = +0.752  CI = [+0.215, +1.229]  POS
  s2: β = −0.120  CI = [−0.798, +0.393]  SPANS_0
  s3: β = −0.457  CI = [−1.031, −0.019]  NEG
  s4: β = −4.115  CI = [−4.863, −3.573]  NEG
```

CI của s0 và s1 **không giao nhau và ngược dấu**. Nên −1.639 ± 2.117 không phải
"một số đo nhiễu" mà là **trung bình của hai regime khác nhau** — báo cáo mean ± std
ở đây sai về bản chất.

**Bằng chứng quyết định rằng đây là calibration chứ không phải thiếu dữ liệu:**
VisDrone `scale_medium` **cũng fail**, với 4,196 event và overlap 0.583. Ở đó không
có vấn đề cỡ mẫu lẫn overlap — chỉ còn calibration regime.

**(b) 35 event** (TT100K: 26) so với 1,966 của COCO — đúng cảnh báo n ≈ 30.

**(c) Overlap yếu (A1) — yếu tố góp phần, không phải nguyên nhân chính.** 17.0% so
với 43–44% ở hai dataset kia. Đáng dẫn để nối hai kết quả, nhưng phải nói đúng vai:
nó làm coefficient bị leverage trên vùng hẹp, khiến giá trị cực đoan hơn — nó không
tạo ra sự bất đồng giữa các seed.

> **Cách vá:** đánh dấu 3 term trên là không diễn giải được ở mức seed mean, **thay
> mean ± std bằng bảng per-seed β + CI** (bimodality tự hiện ra), và viện dẫn cả ba
> lý do theo đúng thứ tự quan trọng: bimodality → event count → overlap (dẫn về A1).

Artefact: `c1_regression_per_seed.csv`, `c1_regression_interpretability.csv`.

---

## C2 — Tách variance: KHÔNG thay được B4, mà biện minh cho B4

**Verdict: phải vá, và kết quả ngược giả thuyết.** Bootstrap image-level đã có sẵn
trong `bootstrap_ci.json` nên phép đo này tốn 0 giờ.

| Dataset | Scale | between-seed sd | within-seed boot sd | tỉ lệ | Nguồn trội | seed std báo thiếu? |
|---|---|---:|---:|---:|---|---|
| COCO | Small | 0.86pp | 0.75pp | 1.2× | comparable | không |
| COCO | Medium | 1.94pp | 0.47pp | 4.1× | calibration | không |
| COCO | Large | 1.97pp | 0.53pp | 3.7× | calibration | không |
| TT100K | Small | 2.37pp | 1.03pp | 2.3× | comparable | không |
| TT100K | Medium | 1.02pp | 0.53pp | 1.9× | comparable | không |
| TT100K | Large | 0.85pp | 1.05pp | **0.8×** | **eval sampling** | **CÓ** |
| VisDrone | Small | 29.25pp | 0.54pp | **54.2×** | **calibration** | không |
| VisDrone | Medium | 19.22pp | 0.64pp | **30.2×** | **calibration** | không |
| VisDrone | Large | 0.35pp | 0.65pp | **0.5×** | **eval sampling** | **CÓ** |

Lấy mẫu eval set góp gần như **không gì** cho VisDrone small/medium — 0.54pp so với
29.25pp. **B4 đáng chạy.** Nhưng phải đổi câu hỏi của B4: chuỗi QFR_S =
[92.8, 24.0, 34.0, 40.6, 90.2] là **bimodal** (s0/s4 một regime, s1/s2/s3 regime
khác), nên câu hỏi không phải "N = 512 có giảm CV không" mà **"N = 512 có xoá được
hai regime không"**.

Và một phát hiện ngược chiều: với **large** trên TT100K (0.8×) và VisDrone (0.5×),
bootstrap sd **vượt** between-seed sd. Ở đó seed std đang **báo thiếu** uncertainty —
đúng các cell n ≈ 26–35 mà checklist đã cảnh báo. Đây là lý do C3 bắt buộc.

Artefact: `c2_variance_decomposition.csv`.

---

## C3 — Bootstrap CI: QFR đã có sẵn, AP cần chạy riêng

**QFR: xong.** Cả 15 run đều đã có CI image-level 1000 replicate. Tổng hợp lên
Table 2:

| Dataset | Scale | QFR | CI image-level (chỉ within-seed) | between-seed sd | CI gộp |
|---|---|---:|---|---:|---|
| COCO | Small | 23.80% | [22.35, 25.27] | 0.86pp | [21.57, 26.02] |
| COCO | Medium | 16.45% | [15.57, 17.42] | 1.94pp | [12.54, 20.36] |
| COCO | Large | 24.46% | [23.44, 25.51] | 1.97pp | [20.47, 28.45] |
| TT100K | Small | 15.29% | [13.30, 17.35] | 2.37pp | [10.22, 20.36] |
| TT100K | Medium | 7.46% | [6.43, 8.52] | 1.02pp | [5.20, 9.72] |
| TT100K | Large | 5.33% | [3.38, 7.48] | 0.85pp | [2.68, 7.97] |
| VisDrone | Small | 56.33% | [55.27, 57.39] | 29.25pp | **[0, 100] clipped** |
| VisDrone | Medium | 28.41% | [27.14, 29.64] | 19.22pp | **[0, 66.10] clipped** |
| VisDrone | Large | 3.17% | [1.95, 4.50] | 0.35pp | [1.73, 4.61] |

> **Cái bẫy phải tránh:** CI image-level gộp qua seed chỉ đo lấy mẫu eval set **trong**
> một run, **không** chứa variance giữa seed. Nếu Table 2 chỉ ghi
> `QFR_S = 56.33% [55.27, 57.39]` cho VisDrone thì đó là con số sai lệch nghiêm trọng
> — khoảng thật phủ gần như toàn dải. Nên `c3_qfr_bootstrap_ci.csv` mang thêm cột
> `between_seed_sd`, CI gộp, và cờ `reporting_requirement`. VisDrone small/medium bị
> đánh `report_per_seed_only`: khi CI gộp tràn ra ngoài [0, 1] thì không có khoảng đơn
> nào diễn giải được, phải liệt kê per-seed. Cùng logic với C1.

**AP theo scale: chưa chạy, cần anh chạy** (xem phần dưới). Em đã viết script và
xác minh cơ chế, nhưng job nhiều giờ không sống qua giữa các lượt của em.

Artefact: `c3_qfr_bootstrap_ci.csv`.

---

## Còn lại chưa đóng

| Mục | Tình trạng | Vì sao |
|---|---|---|
| **C3** AP-by-scale bootstrap CI | **Script xong, cần anh chạy** | Offline, CPU, nhưng nhiều giờ. Xem phần dưới. |
| **B4** calibration-size ablation | **Chưa chạy, và C2 đã biện minh** | Cần GPU. `config.calibration.size = 128`; cần thêm N = 512 × 3 seed trên VisDrone. Repo chỉ có ablation *thành phần* (`int8_small_rich`, `int8_scale_balanced`), không có ablation *cỡ*. |
| **B1** per-channel scheme | **Không xác nhận được** | Engine đã build với profiling verbosity chỉ có layer name. Muốn xác nhận phải build lại engine → cần GPU. Nếu không build lại thì phải ghi vào Limitations. |
| Preprocessing calibration = inference | Chưa kiểm tra tách biệt | Nằm trong `src/calibration.py` / `src/inference.py`, kiểm tra được offline bằng đọc code nếu anh muốn em làm tiếp. |

B4 và per-channel là **hai mục duy nhất còn cần GPU**. Checklist ước tính 1.5 ngày
GPU cho B1+B4; thực tế B1 đã xong, chỉ còn B4 (1 ngày) và tùy chọn build lại engine.

---

## Hướng dẫn chạy AP bootstrap CI

Dùng [scripts/run_bootstrap_ap.sh](scripts/run_bootstrap_ap.sh). Nó lo preflight,
detach, log, và theo dõi tiến độ.

```bash
cd /mnt/d/Project/Quantization

./scripts/run_bootstrap_ap.sh smoke     # 20 replicate, 1 dataset, vài phút
./scripts/run_bootstrap_ap.sh start     # bản thật 1000 replicate
./scripts/run_bootstrap_ap.sh status    # tiến độ, tốc độ, ETA
./scripts/run_bootstrap_ap.sh watch     # bám log; Ctrl-C không ảnh hưởng job
./scripts/run_bootstrap_ap.sh results   # in các khoảng đã xong
./scripts/run_bootstrap_ap.sh stop      # dừng, có bước xác nhận
```

Đổi mặc định qua biến môi trường: `REPLICATES=200 OUTPUT_ID=quick
./scripts/run_bootstrap_ap.sh start`.

Những gì runner xử lý sẵn:

- **Preflight** kiểm tra cả 18 prediction file và 3 dataset root **trước khi** chạy,
  nên thiếu file thì biết ngay thay vì sau khi đã đốt vài giờ.
- **Detach bằng `setsid`** nên đóng terminal không giết job (`nohup` chỉ chặn SIGHUP).
- **Tạo log trước khi launch**, sửa đúng lỗi `tail: cannot open` — trên `/mnt` drvfs
  file vừa redirect có thể chưa hiện ra ngay.
- **`status` tự tìm log qua `/proc/<pid>/fd/1`**, nên đọc được cả job chạy tay.
- **`stop` bắt gõ lại output-id để xác nhận**, và từ chối chạy khi không có terminal.
  Cần bỏ qua trong script thì dùng `FORCE=1`.
- **`start` từ chối** khi đã có job cùng output-id, hoặc khi thư mục kết quả đã tồn tại
  (kèm sẵn lệnh cần chạy để xử lý).

Thứ tự dataset xếp từ nhỏ đến lớn có chủ đích: VisDrone xong trước (~2h) nên anh có
kết quả sớm cho đúng dataset đang có vấn đề, COCO chạy cuối. Mỗi dataset ghi
`intervals.json` ngay khi xong, nên mất điện giữa đường vẫn còn phần đã chạy.

**Ước tính:** VisDrone đo thật 1.30s/replicate → 21.7 phút/run × 6 run ≈ 2.2h.
TT100K nhanh hơn. COCO chậm nhất (5,000 ảnh, 732k prediction). Tổng cỡ overnight. Sau
20 replicate đầu của mỗi run, log tự in tốc độ thật và ETA của run đó — dùng số đó
thay cho ước tính này.

**RAM:** đỉnh ~3–4GB ở COCO (`evalImgs` của một run). Máy 11GB đủ. Chỉ giữ một
evaluator tại một thời điểm.

**Chạy lại:** `bootstrap_ap_ci.py` từ chối ghi đè, nên `rm -rf
outputs/bootstrap_ap/partab_20260814` trước, hoặc dùng `OUTPUT_ID` khác.

### Cơ chế, và tại sao nó đúng

`COCOeval.evaluate()` chạy **một lần** mỗi run (phần đắt), sau đó mỗi replicate chỉ
chạy lại `accumulate()` trên một index đã resample. Định nghĩa AP vẫn là pycocotools
official, không phải bản tự viết — đúng rule 4 trong `AGENTS.md`. Ba kiểm tra đã pass:

| Kiểm tra | Kết quả |
|---|---|
| `maxDets=[100]` cho AP y hệt official đã lưu | diff = `0.00e+00` cho AP, AP_S, AP_M, AP_L |
| Identity resample tái tạo point estimate | khớp tuyệt đối |
| Replicate thật cho spread hợp lý | AP_L 0.529 → 0.541–0.576 |

Resample dựa trên `(bootstrap seed, dataset, replicate)` và **không** phụ thuộc tên
run, nên FP32 và cả 5 seed INT8 thấy **cùng một resample** ở mỗi replicate → ΔAP là
paired thật. Script lưu cả vector AP thô của 1000 replicate
(`replicates_<run>.json`), nên mọi thống kê phái sinh về sau tính lại được mà không
phải chạy lại.

### Sau khi xong

`outputs/bootstrap_ap/partab_20260814/<dataset>/intervals.json` chứa CI cho:
`ap` (từng run), `delta_ap_vs_fp32_paired` (từng seed), và
`delta_ap_vs_fp32_paired` cho `int8_random_seed_mean` — dòng cuối này là cái vào
Table 1. Anh nhắn em, em nối vào bảng và cập nhật báo cáo.

> **Lưu ý khi đọc:** CI này là image-level trong một seed, cùng bản chất với CI của
> QFR. Với **VisDrone small/medium** thì variance giữa seed vẫn trội (C2: 54× và
> 30×), nên ΔAP của hai cell đó **cũng phải kèm between-seed sd**, đừng chỉ ghi CI
> bootstrap — cùng cái bẫy đã nêu ở C3.

---

## Hệ quả lên paper

**Không còn mục nào trong checklist gốc chặn việc viết.** Mục 2, 3, 4 trong PART D —
ba mục nói phải xong trước khi viết — đều đã xong. Việc còn lại là C3 cho cột CI của
Table 1, và nó không chặn phần Method/Mechanism.

| Ô | Trạng thái | Thay đổi so với kế hoạch |
|---|---|---|
| Table 1 (ΔAP/RD) | Chờ C3 | Baseline đã chốt, không đổi số. Thêm cột CI khi AP bootstrap xong |
| Figure 1 (metric disagreement) | Sẵn sàng | Không đổi |
| Figure 2 (QFR + stratified) | Sẵn sàng | Stratified chỉ COCO, kèm `a1_binning_variants` chứng minh làm thô bin không cứu được |
| Table 2 (QFR/QRR) | Sẵn sàng, **mạnh hơn** | Thêm sweep 30 cell (`small_lowest` = 0/30) + CI image-level từ C3 |
| **Table 3 (β_scale)** | **Đổi cách trình bày** | 3 term chuyển sang per-seed β + CI; bỏ mean ± std cho các term đó (C1) |
| **Figure 3** | **Đổi nội dung** | Δs distribution theo scale (survivor + flip) thay stacked bar 4 mode |

Kết luận trong PART E vẫn đúng nguyên văn, và câu thứ hai — câu về tính khả thi của
difficulty control — giờ mạnh hơn theo hai đường: không chỉ "chỉ khả thi khi phân bố
chồng lấp đủ", mà còn **làm thô bin không sửa được** (A1), và **cùng cái thiếu overlap
đó cũng làm regression adjustment không diễn giải được** (C1) — nên hạn chế này chạm
vào cả hai công cụ difficulty control mà paper dùng, không chỉ stratification.

Thêm một câu cho Limitations, đến từ C2:

> Trên VisDrone, phương sai QFR giữa các calibration seed lớn hơn phương sai do lấy
> mẫu tập đánh giá 54× ở small và 30× ở medium, và phân bố theo seed là bimodal; nên
> QFR gộp theo seed của hai cell đó không được diễn giải như một điểm ước lượng duy
> nhất. Ngược lại, ở large trên TT100K và VisDrone, lấy mẫu tập đánh giá lại trội hơn
> (n ≈ 26–35 flip), nên độ lệch giữa seed **báo thiếu** bất định ở đúng các cell đó.

Thêm một câu nên đưa vào Results, đến từ A3 + A4 + B3 gộp lại:

> Quantization dịch phân bố confidence xuống dưới, với độ lớn tăng theo object scale;
> việc dịch chuyển đó có biến thành lost detection hay không lại do khoảng cách từ
> score FP32 tới operating point quyết định. Cùng một cơ chế cho ra ΔAP_L lớn nhất,
> QFR_S lớn nhất, và số false positive giảm — ba quan sát trước đây trông như mâu thuẫn.
