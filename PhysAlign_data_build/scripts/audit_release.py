#!/usr/bin/env python
"""Scan release candidates and their Git index bytes without printing secrets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
                '.ruff_cache', 'build', 'dist', 'data', 'workspaces', 'outputs', 'exports', 'tmp'}
PRIVATE_DIRS = {'data', 'workspaces', 'outputs', 'exports', 'tmp'}
PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    'API token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{25,})\b'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'AWS access key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'Windows user path': re.compile(r'(?i)\b[A-Z]:[/\\]+Users[/\\]+[^/\\\s"<>]+'),
    'Unix home path': re.compile(r'/(?:home|Users|root)/[^/\s"<>]+'),
    'email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}
TEXT_SUFFIXES = {'.py', '.js', '.html', '.css', '.json', '.jsonl', '.md', '.txt', '.toml',
                 '.yaml', '.yml', '.svg', '.ps1', '.sh', '.in', '.cfg', '.ini', '.example'}
TEXT_NAMES = {'.gitignore', '.gitattributes', 'LICENSE', 'NOTICE', 'Dockerfile'}


def _git(root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root), *args], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def candidate_files(root: Path = ROOT) -> list[Path]:
    result = _git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z')
    if result is not None:
        return sorted({root / value.decode('utf-8') for value in result.split(b'\0') if value})
    result_files = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in IGNORED_DIRS and not name.endswith('.egg-info')]
        result_files.extend(Path(directory) / name for name in files)
    return sorted(result_files)


def inspect_content(relative: str, content: bytes) -> list[tuple[str, int, str]]:
    path = Path(relative)
    findings = []
    if path.parts[0] in PRIVATE_DIRS:
        return [(relative, 0, 'generated/private directory is a release candidate')]
    name = path.name.lower()
    if ((name == '.env' or name.startswith('.env.')) and name != '.env.example' or
            path.suffix.lower() in {'.key', '.pem', '.p12', '.pfx'} or
            name.startswith('credentials') and name.endswith('.json') or
            name.endswith('.local.json') or name in {'id_rsa', 'id_ed25519'}):
        return [(relative, 0, 'sensitive filename')]
    if name.endswith(('.log', '.api.json')):
        return [(relative, 0, 'runtime log is a release candidate')]
    if len(content) > 5 * 1024 * 1024:
        return [(relative, 0, 'large file requires explicit release review')]
    example_image = relative.startswith('examples/') and path.suffix.lower() == '.png'
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES and not example_image:
        findings.append((relative, 0, 'unexpected binary or file type'))
    for number, line in enumerate(content.decode('utf-8-sig', errors='replace').splitlines(), 1):
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                if label == 'email address':
                    domain = match.group().rsplit('@', 1)[1].lower()
                    if domain in {'example.com', 'example.org', 'example.net'} or domain.endswith(('.test', '.invalid', '.example')):
                        continue
                findings.append((relative, number, label))
                break
        if name == '.env.example' and line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if re.search(r'(KEY|TOKEN|SECRET|PASSWORD)', key, re.I):
                value = value.strip().strip('\"\x27')
                if value and value not in {'replace_with_your_key', 'your_api_key'}:
                    findings.append((relative, number, 'non-placeholder credential in example'))
    return findings


def audit(root: Path = ROOT) -> tuple[list[Path], list[tuple[str, int, str]]]:
    root = root.resolve()
    files, findings = candidate_files(root), []
    tracked = _git(root, 'ls-files', '-z')
    if tracked is None and any((parent / '.git').exists() for parent in (root, *root.parents)):
        findings.append(('.', 0, 'Git index is unreadable; cannot certify staged release contents'))
    tracked = tracked or b''
    tracked_names = {value.decode('utf-8') for value in tracked.split(b'\0') if value}
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            findings.append((relative, 0, 'linked path is not allowed in release'))
            continue
        if path.is_file():
            findings.extend(inspect_content(relative, path.read_bytes()))
        elif relative not in tracked_names:
            findings.append((relative, 0, 'cannot read release candidate'))
        if relative in tracked_names:
            staged = _git(root, 'show', ':' + relative)
            if staged is not None:
                findings.extend((name + ' [index]', line, label)
                                for name, line, label in inspect_content(relative, staged))
    return files, sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    files, findings = audit(args.root)
    if findings:
        print('Release audit failed. Matched values are intentionally not printed.', file=sys.stderr)
        for relative, line, label in findings:
            print(f'- {relative}:{line}: {label}', file=sys.stderr)
        return 1
    print(f'Release audit passed: {len(files)} candidate files; available Git index also checked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
