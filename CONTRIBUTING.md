# Contributing to Meetru

Thanks for wanting to help! Meetru is a small project by one person, written with AI help (see the README), so please keep changes focused and say what they do and why.

## License of your contribution

Meetru is licensed under **GPL-3.0-or-later**. By contributing you agree that your contribution is licensed under the same terms. You keep the copyright to what you write.

## Sign your commits (Developer Certificate of Origin)

Every commit needs a sign-off line saying that you have the right to submit it under this license:

```
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` adds it for you. The sign-off means you agree to the [Developer Certificate of Origin 1.1](DCO). Use your real name. Pull requests with unsigned commits will be asked to fix them first.

Because every contributor keeps their copyright and signs only the DCO, the license cannot be changed to a different one later without asking each of them (later versions of the GPL are already allowed by "or later").

## What to add, and what not to

- **No media in the repository.** No sounds, pictures or videos, not even small ones: the app keeps the user's media in their own data folder, and the starter content is made in code.
- **No code you cannot share.** Don't paste code from other projects unless its license works with GPL-3.0-or-later; if you do, name the source and the license in your pull request and add it to the credits in the README.
- **Keep names generic.** Don't use real people's names, voices or faces, or other people's trademarks (anime, brands, characters), as feature names or in examples.
- **New libraries.** Say what license they have in the pull request. It must work with GPL-3.0-or-later (the current list is in [docs/research/2026-10-30-license-ip-audit.md](docs/research/2026-10-30-license-ip-audit.md)).
- **AI-assisted contributions are fine** if you have read and tested them and can honestly sign off.
- **Every `.py` file starts with the license header** (`tests/test_license_headers.py` checks it):

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) <year> <your name>
```

## Before you open a pull request

```bash
.venv\Scripts\python -m pytest -q
```

All tests must pass. For anything that touches the camera, Meet or the sound, say how you tried it.
