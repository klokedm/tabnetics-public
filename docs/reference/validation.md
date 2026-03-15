---
title: Validation
nav_order: 10
parent: Reference
---

Validation planning, sharding, suite execution, and campaign-specific job builders.

Package source: [`tabnetics.validation`](https://github.com/klokedm/tabnetics-public/tree/main/src/tabnetics/validation)

## Package overview

Validation planning, suite, and shard execution surface behind ``tabnetics-validation-plan``, ``tabnetics-validation-shard``, and ``tabnetics-validation-suite``; evidence-bearing runs use the HuggingFace bundle as the authoritative operational mirror of the public upstream datasets and default to ``dataset_integrity_policy="error"``.

## Stable exports

- `CORE_PROJECT_ROOT` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L33). Module-level constant exported by the package surface.
- `REPO_ROOT` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L34). Module-level constant exported by the package surface.
- `DEFAULT_MAX_PODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L35). Module-level constant exported by the package surface.
- `VALIDATION_SEEDS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L36). Module-level constant exported by the package surface.
- `class Job` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L40).
- `class BenchmarkProfile` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L48).
- `def build_jobs_legacy(*, include_feature_smoke: bool = True, include_validation_suite: bool = True) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L770).
- `def build_jobs_validation1(*, dataset_shards: int = 8) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1004). Validation-1: full-catalog 5-seed runs for baseline + single-feature toggles.
- `def build_jobs_validation4(*, dataset_shards: int = 6) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1239). Validation-4: 3-profile decomposition + PLS-DA guardrail on 6 pods.
- `def build_jobs_validation5(*, dataset_shards: int = 6, val4_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1416). Validation-5: baseline anchor vs complete B+C+D candidate on 6 pods.
- `def build_jobs_validation6(*, dataset_shards: int = 6, val5_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1567). Validation-6: two-profile signal check on the 67-dataset Val-5 catalog.
- `def build_jobs_validation7(*, dataset_shards: int = 6, val6_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1770). Validation-7: classifier upgrade + continuous shrinkage on the 67-dataset catalog.
- `def build_jobs_validation8(*, dataset_shards: int = 6, val7_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1978). Validation-8: all approved fixes + Stage-2 classifier support on 67 datasets.
- `def build_jobs_validation9(*, dataset_shards: int = 6, val8_root: Optional[Path] = None, runtime_profile: str = 'full') -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2180). Validation-9: comprehensive full-stack run with classifier selection A/B.
- `def build_jobs_validation10(*, dataset_shards: int = 6, val9_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2447). Validation-10: simple-vs-MNPO usage across all pipeline stages.
- `def build_jobs_validation11(*, dataset_shards: int = 4, val10_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2728). Validation-11 v2: 2×2 factorial stage decomposition + extensions.
- `def build_jobs_validation12(*, dataset_shards: int = 6, val11_root: Optional[Path] = None, val10_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L3434). Validation-12: regime-conditional gating comparison.
- `def build_jobs_validation13(*, dataset_shards: int = 8, val12_root: Optional[Path] = None, val11_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L3908). Validation-13: gate fixes + profile expansion + ablation + FS-fraction sweep.
- `VALIDATION14_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4602). Module-level constant exported by the package surface.
- `VALIDATION14_ACTIVATION_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4641). Module-level constant exported by the package surface.
- `VALIDATION14_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4651). Module-level constant exported by the package surface.
- `def build_jobs_validation14(*, dataset_shards: int = 6, val13_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5103). Validation-14 full matrix (20 profiles x 35 datasets x 9 seeds).
- `def build_jobs_validation14_activation_smoke(*, dataset_shards: int = 6, val13_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5120). Validation-14 activation smoke (same profiles, 6 datasets, single seed).
- `VALIDATION15_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5223). Module-level constant exported by the package surface.
- `VALIDATION15_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5225). Module-level constant exported by the package surface.
- `def build_jobs_validation15(*, dataset_shards: int = 6, val14_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5563). Validation-15 focused matrix (9 profiles x 35 datasets x 9 seeds).
- `VALIDATION16_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5625). Module-level constant exported by the package surface.
- `VALIDATION16_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5657). Module-level constant exported by the package surface.
- `VALIDATION17_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5672). Module-level constant exported by the package surface.
- `VALIDATION17_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5674). Module-level constant exported by the package surface.
- `def build_jobs_validation16(*, dataset_shards: int = 9, val15_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5991). Validation-16 focused matrix (11 profiles x 64 datasets x 9 seeds).
- `def build_jobs_validation17(*, dataset_shards: int = 9, val15_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6008). Validation-17 focused matrix: Val-16 profile rerun on the packaged after-FS default.
- `VAL18_FULL64` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6103). Module-level constant exported by the package surface.
- `VAL18_DIAG24` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6105). Module-level constant exported by the package surface.
- `VAL18_SINGLETON_METHODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6133). Module-level constant exported by the package surface.
- `VAL18_CLASSIFIER_UNIVERSE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6175). Module-level constant exported by the package surface.
- `VAL19_ADDED_CLASSIFIERS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6208). Module-level constant exported by the package surface.
- `VAL19_HDLSS_EXTREME_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6218). Module-level constant exported by the package surface.
- `VAL19_HDLSS_EXTREME_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6234). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6236). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6257). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_CPU_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6262). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_CPU_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6265). Module-level constant exported by the package surface.
- `VAL18_CPU_HOSTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6267). Module-level constant exported by the package surface.
- `VAL18_CPU_HOST_CAPACITY_CORES` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6274). Module-level constant exported by the package surface.
- `VAL18_GPU_HOSTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6281). Module-level constant exported by the package surface.
- `VAL18_HOST_WORKER_TARGETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6283). Module-level constant exported by the package surface.
- `VAL19_HOST_WORKER_TARGETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6294). Module-level constant exported by the package surface.
- `VALIDATION18_ANCHORS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6703). Module-level constant exported by the package surface.
- `def build_jobs_validation18_anchors(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6715). Val-18 Family A: 8 anchor/control profiles x 64 datasets x 9 seeds.
- `VALIDATION18_SINGLETONS_PROFILE_IDS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6809). Module-level constant exported by the package surface.
- `def build_jobs_validation18_singletons(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6816). Val-18 Family M: 39 methods x 2 variants (raw/scaffold) x 64 datasets x 5 seeds.
- `VALIDATION18_MNPO_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6919). Module-level constant exported by the package surface.
- `def build_jobs_validation18_mnpo(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6948). Val-18 Family N: 25 MNPO/oracle profiles x 64 datasets x 5 seeds.
- `VALIDATION18_STAGE_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7099). Module-level constant exported by the package surface.
- `def build_jobs_validation18_stage(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7146). Val-18 Families D+P+S(D/P): stage/scaffold matrix x 24 diagnostic datasets x 5 seeds.
- `VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7372). Module-level constant exported by the package surface.
- `def build_jobs_validation18_classifiers(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7406). Val-18 Family C + S04: CPU classifier sweep plus TabPFN-only reruns.
- `VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7681). Module-level constant exported by the package surface.
- `def build_jobs_validation19_classifiers(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7745). Val-19 required surface: 7 singleton diagnostics + 4 pool reruns + 2 oracle-compat controls.
- `VALIDATION18_CLS_ORACLE_WT_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7945). Module-level constant exported by the package surface.
- `def build_jobs_validation18_cls_oracle_wt(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7961). Val-18 Family W: classifier-oracle weighting sweep on DIAG24.
- `def build_jobs_validation13_clf_oracle(*, dataset_shards: int = 4, val12_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L8081). Validation-13 classifier oracle comparison — standalone 2×2 factorial.
- `def main() -> None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L8879).
- `FS_BASE_METHODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L114). Module-level constant exported by the package surface.
- `class LoadedTabularDataset` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L129).
- `class DistributionCase` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L139).
- `class AblationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L150).
- `def extract_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L217). Extract dataset meta-features useful for tier assignment / analysis.
- `CATALOG` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L311). Module-level constant exported by the package surface.
- `DATASET_SETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L312). Module-level constant exported by the package surface.
- `COMPONENT_DEFAULTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L315). Module-level constant exported by the package surface.
- `NOOP_COMPONENTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L341). Module-level constant exported by the package surface.
- `def load_feature_selection_dataset(spec: ValidationDatasetSpec, seed: int, allow_synthetic_fallback: bool, sample_cap: int, feature_cap: int, source_policy: Optional[str] = None, class_integrity_policy: str = 'error', class_min_classes: int = 2, class_min_class_count: int = 1, require_hf_source: bool = False) -> LoadedTabularDataset` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L1269).
- `def resolve_dataset_ids(catalog: Dict[str, ValidationDatasetSpec], dataset_sets: Sequence[str], explicit_ids: Sequence[str], exclude_ids: Sequence[str], pipelines: Sequence[str], max_datasets: Optional[int]) -> List[str]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2291).
- `def build_ablation_configs(profile: str, base_components: Dict[str, bool], pipelines: Sequence[str], constrained_components: Optional[Sequence[str]] = None) -> List[AblationConfig]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2340).
- `def run_validation_suite(args: argparse.Namespace) -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2504).
- `def build_arg_parser() -> argparse.ArgumentParser` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2880).

## Related modules

- `tabnetics.validation.core.shard_runner` - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/core/shard_runner.py). Run a single shard from a generated validation plan, backing ``tabnetics-validation-shard`` and consuming the ``plan_*.json`` / ``shards_*.json`` artifacts emitted by ``tabnetics-validation-plan``.

## Module details

### `tabnetics.validation.__init__`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/__init__.py)

Validation planning, suite, and shard execution surface behind ``tabnetics-validation-plan``, ``tabnetics-validation-shard``, and ``tabnetics-validation-suite``; evidence-bearing runs use the HuggingFace bundle as the authoritative operational mirror of the public upstream datasets and default to ``dataset_integrity_policy="error"``.

No top-level public symbols are exported directly from this module.

### `tabnetics.validation.generate_plan`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py)

Generate a sharded validation plan (up to 20 pods) for ``tabnetics-validation-plan``, aligned with the promoted ``df_stage_position="after_fs"`` runtime and the stricter HuggingFace public-source mirror policy.

- `CORE_PROJECT_ROOT` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L33). Module-level constant exported by the package surface.
- `REPO_ROOT` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L34). Module-level constant exported by the package surface.
- `DEFAULT_MAX_PODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L35). Module-level constant exported by the package surface.
- `VALIDATION_SEEDS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L36). Module-level constant exported by the package surface.
- `class Job` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L40).
- `class BenchmarkProfile` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L48).
- `def build_jobs_legacy(*, include_feature_smoke: bool = True, include_validation_suite: bool = True) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L770).
- `def build_jobs_validation1(*, dataset_shards: int = 8) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1004). Validation-1: full-catalog 5-seed runs for baseline + single-feature toggles.
- `def build_jobs_validation4(*, dataset_shards: int = 6) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1239). Validation-4: 3-profile decomposition + PLS-DA guardrail on 6 pods.
- `def build_jobs_validation5(*, dataset_shards: int = 6, val4_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1416). Validation-5: baseline anchor vs complete B+C+D candidate on 6 pods.
- `def build_jobs_validation6(*, dataset_shards: int = 6, val5_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1567). Validation-6: two-profile signal check on the 67-dataset Val-5 catalog.
- `def build_jobs_validation7(*, dataset_shards: int = 6, val6_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1770). Validation-7: classifier upgrade + continuous shrinkage on the 67-dataset catalog.
- `def build_jobs_validation8(*, dataset_shards: int = 6, val7_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L1978). Validation-8: all approved fixes + Stage-2 classifier support on 67 datasets.
- `def build_jobs_validation9(*, dataset_shards: int = 6, val8_root: Optional[Path] = None, runtime_profile: str = 'full') -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2180). Validation-9: comprehensive full-stack run with classifier selection A/B.
- `def build_jobs_validation10(*, dataset_shards: int = 6, val9_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2447). Validation-10: simple-vs-MNPO usage across all pipeline stages.
- `def build_jobs_validation11(*, dataset_shards: int = 4, val10_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L2728). Validation-11 v2: 2×2 factorial stage decomposition + extensions.
- `def build_jobs_validation12(*, dataset_shards: int = 6, val11_root: Optional[Path] = None, val10_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L3434). Validation-12: regime-conditional gating comparison.
- `def build_jobs_validation13(*, dataset_shards: int = 8, val12_root: Optional[Path] = None, val11_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L3908). Validation-13: gate fixes + profile expansion + ablation + FS-fraction sweep.
- `VALIDATION14_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4602). Module-level constant exported by the package surface.
- `VALIDATION14_ACTIVATION_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4641). Module-level constant exported by the package surface.
- `VALIDATION14_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L4651). Module-level constant exported by the package surface.
- `def build_jobs_validation14(*, dataset_shards: int = 6, val13_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5103). Validation-14 full matrix (20 profiles x 35 datasets x 9 seeds).
- `def build_jobs_validation14_activation_smoke(*, dataset_shards: int = 6, val13_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5120). Validation-14 activation smoke (same profiles, 6 datasets, single seed).
- `VALIDATION15_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5223). Module-level constant exported by the package surface.
- `VALIDATION15_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5225). Module-level constant exported by the package surface.
- `def build_jobs_validation15(*, dataset_shards: int = 6, val14_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5563). Validation-15 focused matrix (9 profiles x 35 datasets x 9 seeds).
- `VALIDATION16_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5625). Module-level constant exported by the package surface.
- `VALIDATION16_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5657). Module-level constant exported by the package surface.
- `VALIDATION17_DATASETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5672). Module-level constant exported by the package surface.
- `VALIDATION17_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5674). Module-level constant exported by the package surface.
- `def build_jobs_validation16(*, dataset_shards: int = 9, val15_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L5991). Validation-16 focused matrix (11 profiles x 64 datasets x 9 seeds).
- `def build_jobs_validation17(*, dataset_shards: int = 9, val15_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6008). Validation-17 focused matrix: Val-16 profile rerun on the packaged after-FS default.
- `VAL18_FULL64` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6103). Module-level constant exported by the package surface.
- `VAL18_DIAG24` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6105). Module-level constant exported by the package surface.
- `VAL18_SINGLETON_METHODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6133). Module-level constant exported by the package surface.
- `VAL18_CLASSIFIER_UNIVERSE` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6175). Module-level constant exported by the package surface.
- `VAL19_ADDED_CLASSIFIERS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6208). Module-level constant exported by the package surface.
- `VAL19_HDLSS_EXTREME_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6218). Module-level constant exported by the package surface.
- `VAL19_HDLSS_EXTREME_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6234). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6236). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6257). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_CPU_OLD` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6262). Module-level constant exported by the package surface.
- `VAL19_HDLSS_MODERATE_CPU_NEW` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6265). Module-level constant exported by the package surface.
- `VAL18_CPU_HOSTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6267). Module-level constant exported by the package surface.
- `VAL18_CPU_HOST_CAPACITY_CORES` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6274). Module-level constant exported by the package surface.
- `VAL18_GPU_HOSTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6281). Module-level constant exported by the package surface.
- `VAL18_HOST_WORKER_TARGETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6283). Module-level constant exported by the package surface.
- `VAL19_HOST_WORKER_TARGETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6294). Module-level constant exported by the package surface.
- `VALIDATION18_ANCHORS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6703). Module-level constant exported by the package surface.
- `def build_jobs_validation18_anchors(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6715). Val-18 Family A: 8 anchor/control profiles x 64 datasets x 9 seeds.
- `VALIDATION18_SINGLETONS_PROFILE_IDS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6809). Module-level constant exported by the package surface.
- `def build_jobs_validation18_singletons(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6816). Val-18 Family M: 39 methods x 2 variants (raw/scaffold) x 64 datasets x 5 seeds.
- `VALIDATION18_MNPO_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6919). Module-level constant exported by the package surface.
- `def build_jobs_validation18_mnpo(*, dataset_shards: int = 9, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L6948). Val-18 Family N: 25 MNPO/oracle profiles x 64 datasets x 5 seeds.
- `VALIDATION18_STAGE_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7099). Module-level constant exported by the package surface.
- `def build_jobs_validation18_stage(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7146). Val-18 Families D+P+S(D/P): stage/scaffold matrix x 24 diagnostic datasets x 5 seeds.
- `VALIDATION18_CLASSIFIERS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7372). Module-level constant exported by the package surface.
- `def build_jobs_validation18_classifiers(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7406). Val-18 Family C + S04: CPU classifier sweep plus TabPFN-only reruns.
- `VALIDATION19_CLASSIFIERS_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7681). Module-level constant exported by the package surface.
- `def build_jobs_validation19_classifiers(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7745). Val-19 required surface: 7 singleton diagnostics + 4 pool reruns + 2 oracle-compat controls.
- `VALIDATION18_CLS_ORACLE_WT_PROFILE_MANIFEST` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7945). Module-level constant exported by the package surface.
- `def build_jobs_validation18_cls_oracle_wt(*, dataset_shards: int = 7, val17_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L7961). Val-18 Family W: classifier-oracle weighting sweep on DIAG24.
- `def build_jobs_validation13_clf_oracle(*, dataset_shards: int = 4, val12_root: Optional[Path] = None) -> List[Job]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L8081). Validation-13 classifier oracle comparison — standalone 2×2 factorial.
- `def main() -> None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/generate_plan.py#L8879).

### `tabnetics.validation.suite`

[Source file](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py)

Unified validation suite for FS, DF, and integrated pipeline benchmarks, backing ``tabnetics-validation-suite`` while enforcing the packaged dataset-integrity defaults and the HuggingFace public-source mirror workflow for evidence-bearing validation datasets.

- `REPO_ROOT` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L66). Module-level constant exported by the package surface.
- `FS_BASE_METHODS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L114). Module-level constant exported by the package surface.
- `class LoadedTabularDataset` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L129).
- `class DistributionCase` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L139).
- `class AblationConfig` (class) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L150).
- `def extract_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L217). Extract dataset meta-features useful for tier assignment / analysis.
- `CATALOG` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L311). Module-level constant exported by the package surface.
- `DATASET_SETS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L312). Module-level constant exported by the package surface.
- `COMPONENT_DEFAULTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L315). Module-level constant exported by the package surface.
- `NOOP_COMPONENTS` (constant) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L341). Module-level constant exported by the package surface.
- `def load_feature_selection_dataset(spec: ValidationDatasetSpec, seed: int, allow_synthetic_fallback: bool, sample_cap: int, feature_cap: int, source_policy: Optional[str] = None, class_integrity_policy: str = 'error', class_min_classes: int = 2, class_min_class_count: int = 1, require_hf_source: bool = False) -> LoadedTabularDataset` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L1269).
- `def resolve_dataset_ids(catalog: Dict[str, ValidationDatasetSpec], dataset_sets: Sequence[str], explicit_ids: Sequence[str], exclude_ids: Sequence[str], pipelines: Sequence[str], max_datasets: Optional[int]) -> List[str]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2291).
- `def build_ablation_configs(profile: str, base_components: Dict[str, bool], pipelines: Sequence[str], constrained_components: Optional[Sequence[str]] = None) -> List[AblationConfig]` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2340).
- `def run_validation_suite(args: argparse.Namespace) -> Path` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2504).
- `def build_arg_parser() -> argparse.ArgumentParser` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L2880).
- `def main() -> None` (function) - [Source](https://github.com/klokedm/tabnetics-public/blob/main/src/tabnetics/validation/suite.py#L3003).

---

> Documentation and webpages on this site are generated from authoritative internal sources using a combination of deterministic rules and generative AI. Errors are possible. Please report issues via [GitHub Discussions](https://github.com/klokedm/tabnetics-public/discussions) or email [marko@tabnetics.org](mailto:marko@tabnetics.org).
