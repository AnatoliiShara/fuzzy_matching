import unittest
from src.company_search import (
    prepare_name, normalize_name, find_company, 
    load_data, build_trees, PrefixTree, Trie
)
import pandas as pd

class TestCompanyFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.opf_data, cls.companies_data = load_data('./data/companies_prefix_opf.csv', './data/edr_ua_short.csv')
        cls.prefix_tree, cls.companies_tree, cls.companies_similar_tree = build_trees(cls.opf_data, cls.companies_data)

    def test_prepare_name(self):
        self.assertEqual(prepare_name('ААА ВІП ТРЕВЕЛ'), 'ааа віп тревел')
        self.assertEqual(prepare_name('«АБВЄ Логістик»'), 'абвє логістик')

    def test_normalize_name(self):
        self.assertEqual(normalize_name('ААА ВІП ТРЕВЕЛ', self.prefix_tree), 'ааа віп тревел')
        self.assertEqual(normalize_name('Товариство з обмеженою відповідальністю "АБСОЛЮТ ГРУП"', self.prefix_tree), 'абсолют груп')

    def test_find_company_exact_match(self):
        result = find_company("ААА ВІП ТРЕВЕЛ", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertIsNotNone(result[3])  # Ensure a company is found
        self.assertEqual(result[6], '40902654')  # Check Company Number

    def test_find_company_partial_match(self):
        result = find_company("ААА ВІП ТРВЕЛ", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertIsNotNone(result[3])  # Company should be found despite the typo
        self.assertEqual(result[6], '40902654')  # Should still match "ААА ВІП ТРЕВЕЛ"

    def test_find_company_no_match(self):
        result = find_company("Неіснуюча Компанія", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertIsNone(result[3])  # No company should be found

    def test_load_data(self):
        opf_data, companies_data = load_data('./data/companies_prefix_opf.csv', './data/edr_ua_short.csv')
        self.assertIsInstance(opf_data, pd.DataFrame)
        self.assertIsInstance(companies_data, pd.DataFrame)
        self.assertGreater(len(opf_data), 0)
        self.assertGreater(len(companies_data), 0)

    def test_prepare_prefix_tree(self):
        prefix_tree, _, _ = build_trees(self.opf_data, self.companies_data)
        self.assertIsInstance(prefix_tree, PrefixTree)
        # Test a known prefix
        result = prefix_tree.find_longest_prefix("товариство з обмеженою відповідальністю")
        self.assertIsNotNone(result[0])

    def test_prepare_company_trees(self):
        _, companies_tree, companies_similar_tree = build_trees(self.opf_data, self.companies_data)
        self.assertIsInstance(companies_tree, PrefixTree)
        self.assertIsInstance(companies_similar_tree, Trie)

    def test_find_company_returns_correct_tuple_length(self):
        result = find_company("АБСОЛЮТ ГРУП", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertEqual(len(result), 7)  # Expecting 7 elements in the returned tuple

    def test_normalize_name_removes_opf(self):
        normalized = normalize_name('Товариство з обмеженою відповідальністю "АБСОЛЮТ ГРУП"', self.prefix_tree)
        self.assertNotIn('товариство', normalized.lower())
        self.assertIn('абсолют груп', normalized.lower())

    def test_find_company_correct_company_id(self):
        result = find_company("АБСОЛЮТ ГРУП", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertEqual(result[6], '35369653')  # Check if returned company ID is correct

    def test_find_company_with_quotes(self):
        result = find_company('«АБВЄ Логістик»', self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertEqual(result[6], '45157420')  # Check if correct company is found despite quotes

    def test_find_company_case_insensitive(self):
        result = find_company("аБсОлЮт ГрУп", self.prefix_tree, self.companies_tree, 
                              self.companies_similar_tree, self.companies_data)
        self.assertEqual(result[6], '35369653')  # Should find regardless of case

if __name__ == '__main__':
    unittest.main()
