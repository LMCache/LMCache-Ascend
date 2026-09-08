# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import re

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GUIDE_ROOT = REPOSITORY_ROOT / "docs" / "recommended_deployment_guide"
SHELL_BLOCK_PATTERN = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL)
NOUNSET_PATTERN = re.compile(r"^\s*set\s+-[a-z]*u[a-z]*(?:\s|$)", re.MULTILINE)
UNSAFE_LD_PRELOAD_PATTERN = re.compile(
    r"^\s*(?:export\s+)?LD_PRELOAD=.*:\$LD_PRELOAD[\"']?\s*$", re.MULTILINE
)
CLONE_BRANCH_PATTERN = re.compile(
    r"\bgit\s+clone\b[^\n]*?(?:-b|--branch)\s+(?P<branch>\S+)"
)
VALID_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")


def _guides():
    return GUIDE_ROOT.rglob("*.md")


def test_nounset_shell_blocks_do_not_expand_unset_ld_preload():
    failures = []
    for guide in _guides():
        contents = guide.read_text(encoding="utf-8")
        for block_number, block in enumerate(
            SHELL_BLOCK_PATTERN.findall(contents), start=1
        ):
            if NOUNSET_PATTERN.search(block) and UNSAFE_LD_PRELOAD_PATTERN.search(
                block
            ):
                location = guide.relative_to(REPOSITORY_ROOT)
                failures.append(f"{location} block {block_number}")

    assert not failures, "unsafe LD_PRELOAD expansion under set -u: " + ", ".join(
        failures
    )


def test_documented_git_clone_branches_use_plain_ascii():
    failures = []
    for guide in _guides():
        contents = guide.read_text(encoding="utf-8")
        for match in CLONE_BRANCH_PATTERN.finditer(contents):
            branch = match.group("branch")
            if not branch.isascii() or VALID_BRANCH_PATTERN.fullmatch(branch) is None:
                failures.append(
                    f"{guide.relative_to(REPOSITORY_ROOT)}: invalid branch {branch!r}"
                )

    assert not failures, "; ".join(failures)
