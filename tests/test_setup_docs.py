"""Keep the copy-paste setup commands and links in sync with the real CLIs.

A new teammate follows docs/QUICKSTART.md literally, so a renamed flag or moved
anchor would break their setup without any other test noticing.
"""
import importlib.util
import re
import shlex
from pathlib import Path

import pytest

from scripts import local_stack
from src.access import benchmark_grants

ROOT = Path(__file__).resolve().parents[1]
GUIDES = [ROOT / 'docs' / 'QUICKSTART.md', ROOT / 'docs' / 'WEB_APP.md']


def shell_commands(path):
    """Yield each command in ```sh blocks, with backslash continuations joined."""
    for block in re.findall(r'```sh\n(.*?)```', path.read_text(), flags=re.S):
        for command in block.replace('\\\n', ' ').splitlines():
            if command.strip():
                yield command.strip()


def stack_commands(path):
    commands = []
    for command in shell_commands(path):
        if not command.startswith('python3 scripts/local_stack.py'):
            continue
        try:
            commands.append(shlex.split(command)[2:])
        except ValueError:
            continue  # first line of a multi-line python -c '...' snippet
    return commands


def anchors(path):
    text = path.read_text()
    ids = set(re.findall(r'<a id="([^"]+)"></a>', text))
    for heading in re.findall(r'^#+ (.+)$', text, flags=re.M):
        ids.add(re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-'))
    return ids


@pytest.mark.parametrize('guide', GUIDES, ids=lambda path: path.name)
def test_documented_local_stack_commands_parse(guide):
    commands = stack_commands(guide)
    assert commands, f'{guide.name} documents no local_stack commands'
    for argv in commands:
        arguments = local_stack.parser().parse_args(argv)
        if arguments.command != 'operator':
            continue
        inner = arguments.arguments[1:] if arguments.arguments[:1] == ['--'] else arguments.arguments
        if inner[:2] != ['python', '-m']:
            continue  # inline python -c snippets are not module CLIs
        module = inner[2]
        assert importlib.util.find_spec(module), f'{guide.name} runs missing module {module}'
        if module == 'src.access.benchmark_grants':
            benchmark_grants.parser().parse_args(inner[3:])


def test_quickstart_covers_every_grant_action_in_order():
    actions = [argv[argv.index('src.access.benchmark_grants') + 1]
               for argv in stack_commands(GUIDES[0]) if 'src.access.benchmark_grants' in argv]
    assert actions == ['prepare', 'dry-run', 'apply', 'revoke']


# Prose docs only; docs/evaluation_prompts are hash-pinned grader inputs, not guides.
ALL_DOCS = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md', ROOT / 'evaluation' / 'README.md',
            ROOT / 'frontend' / 'tests' / 'README.md', ROOT / 'tests' / 'deployment' / 'README.md',
            *sorted((ROOT / 'docs').glob('*.md'))]


@pytest.mark.parametrize('guide', ALL_DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_relative_links_and_anchors_resolve(guide):
    for target in re.findall(r'\]\(([^)\s]+)\)', guide.read_text()):
        if re.match(r'[a-z]+:', target):
            continue
        file_part, _, anchor = target.partition('#')
        linked = (guide.parent / file_part).resolve() if file_part else guide
        assert linked.exists(), f'{guide.name} links to missing {target}'
        if anchor and linked.suffix == '.md':
            assert anchor in anchors(linked), f'{guide.name} links to missing anchor {target}'
