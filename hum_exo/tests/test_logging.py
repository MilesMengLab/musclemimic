"""Checkpoint verbosity belongs to the console sink, not checkpoint behavior."""

import logging
from io import StringIO

from rich.console import Console


def test_checkpoint_info_is_file_only_and_warnings_remain_visible(tmp_path, monkeypatch):
    from hum_exo.logging import checkpoint_console_logging

    root = logging.getLogger()
    stream, notices = StringIO(), StringIO()
    terminal_handler = logging.StreamHandler(stream)
    file_handler = logging.FileHandler(tmp_path / "train.log")
    monkeypatch.setattr(root, "handlers", [terminal_handler, file_handler])
    monkeypatch.setattr(root, "level", logging.INFO)
    names = ["absl", "orbax.checkpoint", "musclemimic.algorithms.common.checkpoint_hooks", "hum_exo.test"]
    loggers = [logging.getLogger(name) for name in names]
    for logger in loggers:
        monkeypatch.setattr(logger, "level", logging.INFO)
        monkeypatch.setattr(logger, "handlers", [])
        monkeypatch.setattr(logger, "propagate", True)
        monkeypatch.setattr(logger, "disabled", False)
    try:
        with checkpoint_console_logging(Console(file=notices, color_system=None)):
            loggers[0].info("Initiated CompositeCheckpointHandler.async_save")
            loggers[1].info("Renaming temporary checkpoint directories")
            loggers[2].info(
                "Saved checkpoint (sync) at update 77: /tmp/checkpoint_77",
                extra={
                    "checkpoint_event": "saved",
                    "checkpoint_path": "/tmp/checkpoint_77",
                    "checkpoint_timestep": 1234,
                },
            )
            loggers[2].info("Global timestep: 1,234")
            loggers[0].warning("checkpoint warning must stay visible")
            loggers[1].error("checkpoint error must stay visible")
            loggers[3].info("unrelated training information")
        assert "async_save" not in stream.getvalue()
        assert "Renaming" not in stream.getvalue()
        assert "Global timestep" not in stream.getvalue()
        assert "Saved checkpoint" not in stream.getvalue()
        assert "checkpoint warning must stay visible" in stream.getvalue()
        assert "checkpoint error must stay visible" in stream.getvalue()
        assert "unrelated training information" in stream.getvalue()
        assert notices.getvalue().count("模型已保存") == 1
        assert "checkpoint_77" in notices.getvalue() and "1,234" in notices.getvalue()
        file_handler.flush()
        text = (tmp_path / "train.log").read_text()
        assert "async_save" in text and "Renaming" in text and "Saved checkpoint" in text
        assert not terminal_handler.filters and not file_handler.filters
        assert root.handlers == [terminal_handler, file_handler]
        loggers[0].info("verbosity restored after training")
        assert "verbosity restored" in stream.getvalue()
    finally:
        file_handler.close()


def test_async_checkpoint_notice_reports_submission_not_completion(monkeypatch):
    from hum_exo.logging import checkpoint_console_logging

    root = logging.getLogger()
    stream = StringIO()
    handler = logging.StreamHandler(StringIO())
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(root, "level", logging.INFO)
    logger = logging.getLogger("musclemimic.algorithms.common.checkpoint_hooks")
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    with checkpoint_console_logging(Console(file=stream, color_system=None)):
        logger.info(
            "Saved checkpoint (async)",
            extra={"checkpoint_event": "queued", "checkpoint_path": "/tmp/checkpoint_1", "checkpoint_timestep": 128},
        )
    assert "已提交模型保存" in stream.getvalue()
    assert "模型已保存" not in stream.getvalue()


def test_real_orbax_save_and_restore_remain_intact_with_quiet_console(tmp_path, monkeypatch):
    import orbax.checkpoint as ocp

    from hum_exo.logging import checkpoint_console_logging

    root = logging.getLogger()
    stream = StringIO()
    terminal_handler = logging.StreamHandler(stream)
    file_handler = logging.FileHandler(tmp_path / "orbax.log")
    monkeypatch.setattr(root, "handlers", [terminal_handler, file_handler])
    monkeypatch.setattr(root, "level", logging.INFO)
    monkeypatch.setattr(logging.getLogger("absl"), "level", logging.INFO)
    checkpointer = ocp.Checkpointer(ocp.JsonCheckpointHandler())
    path = tmp_path / "checkpoint"
    expected = {"steps": 8192, "marker": "checkpoint data preserved"}
    try:
        with checkpoint_console_logging(Console(file=stream, color_system=None)):
            checkpointer.save(path, args=ocp.args.JsonSave(expected))
            assert checkpointer.restore(path) == expected
            checkpointer.close()
        file_handler.flush()
        assert "Renaming" not in stream.getvalue()
        assert "Saving checkpoint" not in stream.getvalue()
        assert (tmp_path / "orbax.log").stat().st_size > 0
    finally:
        file_handler.close()
