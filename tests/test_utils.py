"""Tests for configuration and logging utilities."""

import logging
import sys

import pytest
import yaml

from ndt.core.tracker import HighFrequencyTracker
from ndt.utils import load_config
from ndt.utils import save_config
from ndt.utils import setup_logger
from ndt.utils.config import get_default_config


class TestConfig:
    def test_save_and_load_roundtrip(self, tmp_path, capsys):
        config = {
            "sampling_frequency": 10,
            "enable_jump_detection": True,
            "jump_z_threshold": 2.5,
            "layers": ["a", "b"],
        }
        path = tmp_path / "cfg.yaml"
        save_config(config, str(path))

        assert "Saved configuration" in capsys.readouterr().out
        assert load_config(str(path)) == config
        # Keys are written in insertion order (sort_keys=False), block style.
        text = path.read_text()
        assert text.index("sampling_frequency") < text.index("jump_z_threshold")
        assert "{" not in text

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(tmp_path / "missing.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("a: [1, 2\n")
        with pytest.raises(yaml.YAMLError):
            load_config(str(path))

    def test_default_config(self):
        assert get_default_config() == {
            "sampling_frequency": 1,
            "enable_jump_detection": True,
            "jump_window_size": 50,
            "jump_z_threshold": 3.0,
        }

    def test_default_config_is_valid_tracker_kwargs(self, small_mlp):
        tracker = HighFrequencyTracker(small_mlp, **get_default_config())
        try:
            assert tracker.sampling_frequency == 1
            assert tracker.jump_detector.window_size == 50
            assert tracker.jump_detector.z_threshold == 3.0
        finally:
            tracker.close()


@pytest.fixture
def clean_logger():
    names = []

    def _make(name, **kwargs):
        names.append(name)
        return setup_logger(name, **kwargs)

    yield _make
    for name in names:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []


class TestLogging:
    def test_default_logger(self, clean_logger):
        logger = clean_logger("ndt.test.default")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stdout
        assert handler.level == logging.INFO
        assert handler.formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_repeated_setup_does_not_duplicate_handlers(self, clean_logger):
        clean_logger("ndt.test.repeat")
        logger = clean_logger("ndt.test.repeat")
        assert len(logger.handlers) == 1

    def test_file_handler_and_custom_format(self, tmp_path, clean_logger):
        log_file = tmp_path / "run.log"
        logger = clean_logger(
            "ndt.test.file",
            level=logging.DEBUG,
            log_file=str(log_file),
            format_string="%(levelname)s|%(message)s",
        )
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2
        logger.debug("hello debug")
        for handler in logger.handlers:
            handler.flush()
        assert log_file.read_text().strip() == "DEBUG|hello debug"

    def test_level_filters_messages(self, tmp_path, clean_logger):
        log_file = tmp_path / "warn.log"
        logger = clean_logger("ndt.test.level", level=logging.WARNING, log_file=str(log_file))
        logger.info("hidden")
        logger.warning("shown")
        for handler in logger.handlers:
            handler.flush()
        content = log_file.read_text()
        assert "shown" in content
        assert "hidden" not in content
