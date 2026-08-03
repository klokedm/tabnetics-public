from pathlib import Path

from tabnetics.validation.core.shard_runner import _find_done_marker, _skip_done_status_dirs


def test_skip_done_uses_backup_status_dirs(tmp_path, monkeypatch):
    root_out = tmp_path / "tabnetics" / "run_artifacts" / "validation-18" / "val18" / "singletons"
    primary_status = root_out / "_status"
    primary_status.mkdir(parents=True)

    backup_status = tmp_path / "tabnetics.bak.1" / "run_artifacts" / "validation-18" / "val18" / "singletons" / "_status"
    backup_status.mkdir(parents=True)

    safe_id = "val18_singletons__M_RAW_hsic_lasso__ds09"
    marker = backup_status / f"{safe_id}.DONE.ok"
    marker.touch()

    monkeypatch.setenv("SKIP_DONE_STATUS_DIRS", str(backup_status))

    status_dirs = _skip_done_status_dirs(root_out)
    found = _find_done_marker(status_dirs, safe_id)

    assert status_dirs[0] == primary_status
    assert backup_status in status_dirs
    assert found == marker
