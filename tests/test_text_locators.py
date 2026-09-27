"""Exact public text references from legacy and hashed-quote exports."""
from copy import deepcopy
import unittest

from physalign.dataset import PublicDataset
from physalign.storage import digest


class TextLocatorTests(unittest.TestCase):
    def setUp(self):
        self.parts = {'Original statement': '球 A 与球 B。', 'Original question': 'Which ball?', 'Original answer options': '[]'}
        self.locator = {'kind': 'text', 'section': 'stem', 'start': 5, 'end': 8, 'quote': '球 B',
                        'section_sha256': digest(self.parts['Original statement'].encode('utf-8'))}

    def test_unicode_offsets_and_hash_are_validated_without_mutation(self):
        before = deepcopy(self.locator)
        PublicDataset._validate_locator(self.locator, [], self.parts)
        self.assertEqual(self.locator, before)

    def test_legacy_contract_still_works(self):
        PublicDataset._validate_locator({'kind': 'text', 'section': 'statement', 'start': 5, 'end': 8, 'text': '球 B'}, [], self.parts)

    def test_query_source_mapping(self):
        locator = {'kind': 'text', 'section': 'query', 'start': 0, 'end': 5, 'quote': 'Which',
                   'section_sha256': digest(self.parts['Original question'].encode('utf-8'))}
        PublicDataset._validate_locator(locator, [], self.parts)

    def test_modified_section_rejected_even_if_span_matches(self):
        self.parts['Original statement'] += ' extra'
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            PublicDataset._validate_locator(self.locator, [], self.parts)

    def test_incorrect_quote_rejected(self):
        self.locator['quote'] = '球 A'
        with self.assertRaisesRegex(ValueError, 'Text span'):
            PublicDataset._validate_locator(self.locator, [], self.parts)

    def test_bad_offsets_rejected(self):
        for start, end in [(True, 8), (-1, 8), (8, 5), (5, 500)]:
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, 'offsets'):
                PublicDataset._validate_locator({**self.locator, 'start': start, 'end': end}, [], self.parts)

    def test_unknown_mixed_or_missing_fields_rejected(self):
        for locator in [{**self.locator, 'unexpected': 1}, {**self.locator, 'text': '球 B'},
                        {k: v for k, v in self.locator.items() if k != 'section_sha256'}]:
            with self.subTest(locator=locator), self.assertRaisesRegex(ValueError, 'Unsupported exact-text'):
                PublicDataset._validate_locator(locator, [], self.parts)


if __name__ == '__main__':
    unittest.main()
