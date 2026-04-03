# Person 3 Handover For Person 4

## 1. Scope Completed

Person 3 da hoan thanh phan:
- Cross-patient hypergraph grouping
- Multi-granularity evidence selection
- Group-aware feedback tu group context ve selector
- Interpretable fusion diagnostics
- Attention/export cho analysis
- Ablation hooks toi thieu
- Test cho selector va fusion

Pham vi file chinh:
- `src/models/history_selector.py`
- `src/models/fusion.py`
- `src/graph/group_encoder.py`
- `src/graph/hypergraph_builder.py`
- `src/graph/hypergraph_layers.py`
- `src/explainability/attention_export.py`
- `src/evaluation/evaluate_ablation.py`

Ngoai pham vi:
- decoder / joint drug recommendation head
- training loop tong
- DDI objective
- retrieval backbone
- data pipeline

## 2. Stable Outputs For Person 4

Person 4 nen consume output tu `RetrievalEvidenceFusionModel`.

Các key on dinh de dung downstream:

- `fused_repr`
  - Shape: `(B, H)`
  - Day la input chinh cho decoder / drug prediction head cua Person 4.

- `fusion_weights`
  - Shape: `(B, 4)`
  - Branch order: `current`, `self`, `neighbor`, `group`

- `branch_mask`
  - Shape: `(B, 4)`
  - Cho biet branch nao available

- `self_history_context`
  - Shape: `(B, H)`

- `neighbor_history_context`
  - Shape: `(B, H)`

- `group_context`
  - Shape: `(B, H)` hoac `None`

- `evidence_metadata`
  - Metadata giai thich selection va group-aware reweight

- `group_metadata`
  - Metadata giai thich group/hypergraph branch

## 3. What Person 3 Already Handles

Upstream da xu ly xong cac phan sau, Person 4 khong can lam lai:

- Self-history evidence selection
- Neighbor-history evidence selection
- Attribute-level gating truoc visit-level aggregation
- Group-aware reweight tu `group_context` quay lai selector
- Fusion cua `current/self/neighbor/group`
- Fusion diagnostics de tranh branch collapse
- Export metadata cho explainability / ablation

## 4. Important Metadata Available

### In `evidence_metadata`

Visit-level / selection:
- `self_history_weights`
- `neighbor_weights`
- `self_history_scores`
- `neighbor_scores`
- `self_history_content_scores`
- `neighbor_content_scores`
- `self_history_selected_mask`
- `neighbor_selected_mask`
- `self_history_selected_count`
- `neighbor_selected_count`
- `self_history_available_mask`
- `neighbor_available_mask`

Attribute-level:
- `attribute_order`
- `attribute_weights`
- `attribute_scores`
- `self_attribute_weights`
- `self_attribute_scores`
- `neighbor_attribute_weights`
- `neighbor_attribute_scores`
- `self_attribute_mask`
- `neighbor_attribute_mask`
- `self_attribute_available_mask`
- `neighbor_attribute_available_mask`
- `self_attribute_fallback_mask`
- `neighbor_attribute_fallback_mask`

Group-aware:
- `group_influence`
- `group_reweight_scores`
- `group_aware_selection_used`
- `group_available_mask`
- `self_group_influence`
- `neighbor_group_influence`
- `self_group_reweight_scores`
- `neighbor_group_reweight_scores`

### In fusion output

- `branch_entropy`
- `normalized_branch_entropy`
- `dominant_branch_index`
- `dominant_branch_name`
- `dominant_branch_weight`
- `branch_collapse_flag`
- `branch_collapse_score`
- `branch_balance_score`
- `branch_balance_gap`
- `branch_contribution_norms`
- `fusion_entropy_loss`
- `fusion_balance_loss`

## 5. Expected Usage By Person 4

Person 4 nen dung:

- `fused_repr` lam input chinh cho joint drug recommendation head
- `fusion_weights` neu can branch-aware logging hoac auxiliary conditioning
- `evidence_metadata` neu muon giu explainability downstream
- `fusion_entropy_loss` va `fusion_balance_loss` neu muon trainer ngoai dung lam auxiliary loss

Person 4 khong nen:
- tinh lai selector
- tinh lai hypergraph grouping
- tinh lai fusion
- thay doi contract metadata cua Person 3 neu khong co bug integration ro rang

## 6. Data Flow

```text
batch
-> encoder
-> group_encoder
-> history_selector
-> fusion_module
-> fused_repr + fusion_weights + evidence_metadata + group_metadata
-> Person 4 decoder / drug prediction head
```

## 7. Assumptions / Fallbacks

- Neu upstream chua co tensor attribute-specific chi tiet, selector dung fallback learned projection tu representation hien co.
- `diagnosis`, `procedure`, `lab`, `vital` la bat buoc trong attribute gating.
- `medication` duoc ho tro neu representation hien tai cho phep.
- `group_context` co the khong co; pipeline van chay duoc.
- Metadata moi la additive, khong thay the field cu.
- Auxiliary fusion losses dang duoc expose ra output, chua tu noi vao training loop tong.

## 8. Verification

Run:

```powershell
pytest tests\test_history_selector.py tests\test_fusion.py -q
```

Current status:
- `10 passed`

Relevant tests:
- `tests/test_history_selector.py`
- `tests/test_fusion.py`
