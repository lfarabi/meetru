# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_python_file_names_its_license_and_owner():
    files = [*ROOT.glob("*.py"), *(ROOT / "tests").glob("*.py")]
    missing = [f.name for f in files
               if "SPDX-License-Identifier: GPL-3.0-or-later" not in f.read_text(encoding="utf-8")[:300]
               or "Copyright (C)" not in f.read_text(encoding="utf-8")[:300]]
    assert len(files) > 15 and not missing, missing
