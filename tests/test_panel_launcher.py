"""Validate user-supplied GPU allocation before launching model processes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_panel import ROOT, jobs, load_panel


class PanelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'panel.json'
        self.panel = json.loads((ROOT / 'configs/panel.example.json').read_text(encoding='utf-8'))

    def load(self, panel):
        self.path.write_text(json.dumps(panel), encoding='utf-8')
        return load_panel(self.path)

    def test_supplied_gpu_groups_and_resume_reach_child_command(self):
        panel = {'jobs': [dict(self.panel['jobs'][0], gpus=[3, 5])]}
        output = Path(self.temp.name) / 'runs'
        run = output / panel['jobs'][0]['name']
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text('{}')
        name, devices, command = list(jobs('study', output, self.load(panel), resume=True))[0]
        self.assertEqual(name, panel['jobs'][0]['name'])
        self.assertEqual(devices, '3,5')
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--output') + 1], str(run.resolve()))

    def test_overlapping_or_invalid_gpu_groups_are_rejected(self):
        for devices in ([0, 1], [], [3, 3], [-1], [True], ['3']):
            with self.subTest(devices=devices):
                panel = copy.deepcopy(self.panel)
                panel['jobs'][1]['gpus'] = devices
                with self.assertRaisesRegex(ValueError, 'GPU indices'):
                    self.load(panel)

    def test_unsafe_names_and_missing_configs_are_rejected(self):
        for key, value in (('name', '../escape'), ('name', self.panel['jobs'][0]['name']),
                           ('config', 'missing-model.json')):
            with self.subTest(key=key, value=value):
                panel = copy.deepcopy(self.panel)
                panel['jobs'][1][key] = value
                with self.assertRaises(ValueError):
                    self.load(panel)
