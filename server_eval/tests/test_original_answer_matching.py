import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server_eval.match_original_answers import (_reference, damaged_latex_commands, display_text,
                                               statement_match, upstream_images)


class MatchingTests(unittest.TestCase):
    def test_display_preserves_math_commands_and_renders_escaped_paragraphs(self):
        value = r'Given $\nu=2$ and $\nabla f\neq0$.\n\nA. First\n\nB. Second\n'
        text = display_text(value)
        self.assertIn(r'\nu=2', text)
        self.assertIn(r'\nabla f\neq0', text)
        self.assertIn('\n\nA. First\n\nB. Second', text)
        self.assertFalse(text.endswith(r'\n'))

    def test_only_known_format_transform_is_accepted_not_fuzzy_physics(self):
        upstream = r'A dielectric with $\varepsilon \neq 1$.\n\nFind the speed.'
        local = upstream.replace('\\n', '\n')
        self.assertEqual(statement_match(local, upstream), 'identified_legacy_backslash_n_decode')
        self.assertEqual(damaged_latex_commands(local, upstream), {r'\neq': 1})
        self.assertIsNone(statement_match('The mass is 2 kg.', 'The mass is 3 kg.'))

    def test_ordinary_literal_line_breaks_are_not_called_latex_damage(self):
        self.assertEqual(damaged_latex_commands('A. Test\nB. Next', r'A. Test\nB. Next'), {})

    def test_image_markup_removal_does_not_remove_question_subparts(self):
        text = 'Context\n<image_1>\n(a) Find x. ![](images/a.jpg)\n(b) Find y.'
        self.assertEqual(display_text(text), 'Context\n\n(a) Find x. \n(b) Find y.')

    def test_reference_choice_set_is_not_multiple_individually_accepted_choices(self):
        key = _reference({'question_type': 'Multiple Choice', 'answer': ['ACD'], 'solution': ''}, 'live2603', 'fixture')
        self.assertEqual((key['kind'], key['accepted']), ('choice_set', ['A', 'C', 'D']))

    def test_open_physics_and_proofs_do_not_use_string_equality_or_guess_units(self):
        key = _reference({'answer': 'A', 'reasoning': 'Ampere unit, not option A'}, 'seephys', 'fixture')
        self.assertEqual(key['kind'], 'human')
        key = _reference({'final_answer': ['2.47'], 'solution': [], 'unit': 'm', 'error': '1e-1'}, 'olympiad', 'fixture')
        self.assertEqual(key['kind'], 'human')
        self.assertIn('official_tolerance', key['reference_answer'])

    def test_solution_images_are_never_part_of_question(self):
        image = {'src': 'fixture'}
        self.assertEqual(upstream_images({'images': [image], 'solution_images': [{'src': 'secret'}]}, 'physelite'), [image])


if __name__ == '__main__':
    unittest.main()
