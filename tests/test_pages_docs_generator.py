"""Tests for the GitHub Pages docs generator."""

from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs import generate_pages_site as mod


def test_build_site_tree_contains_expected_pages():
    """Generator emits the expected Pages structure."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)

    expected = {
        "CNAME",
        "_config.yml",
        "_includes/head_custom.html",
        "assets/css/custom.scss",
        "index.md",
        "USING.md",
        "BACKGROUND.md",
        "RESULTS.md",
        "TABARENA_RESULTS.md",
        "how-it-works.md",
        "reference/index.md",
        "reference/pipeline.md",
        "reference/feature-selection.md",
        "reference/validation.md",
    }
    assert expected.issubset(site)
    assert site["CNAME"] == "tabnetics.org\n"
    assert "remote_theme: just-the-docs/just-the-docs" in site["_config.yml"]


def test_synced_docs_have_front_matter_and_public_links():
    """Synced markdown pages add front matter and rewrite public links."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)

    index_doc = site["index.md"]
    using_doc = site["USING.md"]
    background_doc = site["BACKGROUND.md"]
    results_doc = site["RESULTS.md"]

    assert index_doc.startswith("---\ntitle: Home\nnav_order: 1\nnav_exclude: false\n---\n")
    assert "[Using Tabnetics](USING.md)" in index_doc
    assert f"[Apache 2.0]({mod.PUBLIC_BLOB}/LICENSE)" in index_doc
    assert 'pip install "tabnetics[feature-selection-optional,benchmarks]"' in index_doc
    assert "lean core-only package" in index_doc
    assert "## What's new in 0.5.0" in index_doc
    assert "55,117 successful runs" in index_doc
    assert "Optional integrations and benchmark backends can rely on third-party libraries with separate licenses and use terms." in index_doc
    assert "[Third-party integrations and licenses](USING.md#third-party-integrations-and-licenses)" in index_doc
    assert "## Operational defaults" in index_doc
    assert "tabnetics-validation-plan" in index_doc
    assert "Adaptive game playing using multiplicative weights" in index_doc

    assert using_doc.startswith("---\ntitle: Using Tabnetics\nnav_order: 2\n---\n")
    assert f"[LICENSE]({mod.PUBLIC_BLOB}/LICENSE)" in using_doc
    assert 'pip install "tabnetics[feature-selection-optional,benchmarks]"' in using_doc
    assert "expanded feature-selection and benchmark/backend library set immediately" in using_doc
    assert "## Third-party integrations and licenses" in using_doc
    assert "This table covers the direct optional integrations surfaced by the current public docs/code paths." in using_doc
    assert "default TabPFN-2.5 weights are non-commercial" in using_doc
    assert "Prior Labs License (Apache 2.0 with additional attribution)" in using_doc
    assert "[SHAP](https://pypi.org/project/shap/)" in using_doc
    for package_name in (
        "boruta",
        "shap",
        "pyvinecopulib",
        "mapie",
        "datasets",
        "flaml",
        "lightgbm",
        "xgboost",
        "catboost",
        "tabpfn",
    ):
        assert f"(`{package_name}`)" in using_doc
    assert "## Validation campaigns" in using_doc
    assert "## Reproducibility and data source policy" in using_doc
    assert "## Uncertainty and conformal outputs" in using_doc
    assert "tabnetics-validation-shard" in using_doc
    assert "classifier-conformal-method" in using_doc

    assert "Their upstream licenses/terms still apply when those integrations are enabled" in background_doc
    assert "rifkin04a" in background_doc.lower()
    assert "10.1016/0022-2496(75)90001-2" in background_doc
    assert "10.1214/08-STS275" in background_doc

    assert "combined validation campaigns with **55,117** successful runs across **210** pipeline profiles" in results_doc
    assert "| Datasets with BA ≥ 0.90 | 31 / 63 |" in results_doc
    assert "| SOTA comparison: above / within / below | 33 / 19 / 11 |" in results_doc
    assert "| Val-19 bridge (V) | 6 | Matched FULL64 regime-pool and oracle-control reruns |" in results_doc
    assert "10.1038/89044" in results_doc
    assert "assets/images/sota_comparison.png" in results_doc
    assert "figures/public/" not in results_doc

    assert "authoritative internal sources" in index_doc
    assert mod.DISCUSSIONS_URL in index_doc
    assert f"mailto:{mod.AUTHOR_EMAIL}" in using_doc


def test_reference_pages_include_source_links_and_public_symbols():
    """AST-derived reference pages include stable symbols and public source URLs."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)

    pipeline_page = site["reference/pipeline.md"]
    validation_page = site["reference/validation.md"]
    benchmarks_page = site["reference/benchmarks.md"]

    assert "`class DistributionFeatureSelectionPipeline`" in pipeline_page
    assert "`class DFFSConfig`" in pipeline_page
    assert "load_df_fs_model_bundle(path: str)" in pipeline_page
    assert "src/tabnetics/pipeline/pipeline.py#L" in pipeline_page
    assert 'df_stage_position="after_fs"' in pipeline_page

    assert "build_jobs_validation17" in validation_page
    assert "run_validation_suite" in validation_page
    assert "src/tabnetics/validation/generate_plan.py#L" in validation_page
    assert "tabnetics-validation-shard" in validation_page
    assert "tabnetics.validation.core.shard_runner" in validation_page
    assert "public upstream datasets" in validation_page
    assert "tabnetics-benchmark" in benchmarks_page
    assert "validation-catalog data policy" in benchmarks_page
    assert "BootstrapGOFSelector" in site["reference/distribution.md"]
    assert "FeatureSelectionResult" in site["reference/feature-selection.md"]
    assert "`bio` (module)" in site["reference/domains.md"]
    assert "`face` (module)" in site["reference/domains.md"]


def test_all_exports_from_package_dunders_are_rendered():
    """Each package page should mention every name declared in package __all__."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)

    for page in mod.PACKAGE_PAGES:
        exports = mod.extract_package_exports(repo_root, page.package)
        if not exports:
            continue
        page_text = site[f"reference/{page.slug}.md"]
        for export in exports:
            assert export in page_text, (page.package, export)


def test_every_generated_markdown_page_gets_the_canonical_footer():
    """All generated markdown pages should carry the same footer once."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)
    footer_phrase = "Documentation and webpages on this site are generated from authoritative internal sources"

    for rel_path, content in site.items():
        if not rel_path.endswith(".md"):
            continue
        assert content.count(footer_phrase) == 1, rel_path
        assert mod.DISCUSSIONS_URL in content, rel_path
        assert f"mailto:{mod.AUTHOR_EMAIL}" in content, rel_path


def test_compare_site_tree_detects_drift(tmp_path):
    """Check mode should detect stale or modified generated files."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)
    output_dir = tmp_path / "docs"

    mod.write_site_tree(site, output_dir)
    assert mod.compare_site_tree(site, output_dir) == []

    (output_dir / "index.md").write_text("stale\n", encoding="utf-8")
    problems = mod.compare_site_tree(site, output_dir)
    assert problems
    assert any("current/index.md" in problem for problem in problems)


def test_checked_in_docs_match_generator_output():
    """The committed docs tree should stay in sync with the generator output."""

    repo_root = Path(__file__).resolve().parents[2]
    site = mod.build_site_tree(repo_root)
    assert mod.compare_site_tree(site, repo_root / "docs") == []


def _slugify_heading(text: str) -> str:
    """Approximate GitHub/Jekyll heading anchors for local-link validation."""

    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def test_core_source_docs_have_valid_local_links():
    """Source markdown should resolve local files and heading anchors in-repo."""

    repo_root = Path(__file__).resolve().parents[2]
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    docs = [
        repo_root / "core/README.md",
        repo_root / "core/USING.md",
        repo_root / "core/BACKGROUND.md",
        repo_root / "core/RESULTS.md",
        repo_root / "core/TABARENA_RESULTS.md",
    ]

    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        for _label, target in link_re.findall(text):
            if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
                continue

            rel_target, _, anchor = target.partition("#")
            resolved = (doc_path.parent / rel_target).resolve() if rel_target else doc_path
            assert resolved.exists(), (doc_path, target)

            if not anchor:
                continue

            target_text = resolved.read_text(encoding="utf-8")
            anchors = {
                _slugify_heading(match.group(2))
                for match in heading_re.finditer(target_text)
            }
            assert anchor in anchors, (doc_path, target, sorted(anchors))
