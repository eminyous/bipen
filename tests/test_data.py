import inspect
import re

import pytest

from bipen import plots, tables
from bipen.data import File, require


def files_read(module):
    """The data files a module names, as File.NAME or File.bounds_lasserre_size(m)."""
    source = inspect.getsource(module)
    names = {getattr(File, name) for name in re.findall(r"File\.([A-Z_]+)\b", source)}
    names |= {File.bounds_lasserre_size(int(m)) for m in re.findall(r"File\.bounds_lasserre_size\((\d+)\)", source)}
    return names


def test_tables_check_every_file_they_read():
    assert files_read(tables) - set(tables.NEEDS) == {File.BOUNDS_UNBALANCED}


def test_plots_check_every_file_they_read():
    assert files_read(plots) == {File.POLYA_LEVELS}


def test_require_names_missing_and_empty_files(tmp_path):
    (tmp_path / "polya").mkdir()
    (tmp_path / File.POLYA_LEVELS).write_text("graph,level\n")
    with pytest.raises(SystemExit) as stop:
        require(tmp_path, [File.BOUNDS_SAMPLED, File.POLYA_LEVELS, File.bounds_lasserre_size(20)])
    message = str(stop.value)
    assert "sampled.csv is missing: bipen run bounds/sampled" in message
    assert "levels.csv has no rows: bipen run polya/levels" in message
    assert "lasserre_20.csv is missing: bipen run bounds/lasserre --size 20 --seeds 3" in message


def test_build_writes_nothing_when_data_is_missing(tmp_path):
    with pytest.raises(SystemExit):
        tables.build(tmp_path / "data", tmp_path / "tables")
    assert not (tmp_path / "tables").exists()
