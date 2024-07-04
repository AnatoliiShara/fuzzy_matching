import re
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
import unittest

def prepare_name(name):
    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\s]', ' ', name)  # Replace special characters with space
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with a single space
    return name.strip().lower()

def normalize_name(name, prefix_tree):
    name_parts = name.split()
    normalized_name = []
    for part in name_parts:
        if part not in prefix_tree:
            normalized_name.append(part)
    return ' '.join(normalized_name)

def find_company(name, prefix_tree, companies_tree, companies_similar_tree, companies_data, threshold):
    normalized_name = normalize_name(prepare_name(name), prefix_tree)
    # Perform search (this is a placeholder for the actual search logic)
    for index, row in companies_data.iterrows():
        company_name = row['Company Name']
        if normalized_name in prepare_name(company_name):
            return row.values.tolist()  # Assuming the result is the row itself
    return None

def load_data(opf_file, companies_file):
    opf_data = pd.read_csv(opf_file)
    companies_data = pd.read_csv(companies_file)
    return opf_data, companies_data

def build_trees(opf_data, companies_data):
    prefix_tree = defaultdict(list)
    for index, row in opf_data.iterrows():
        prefix_tree[row['prefix'].lower()] = row['norm'].lower()
    companies_tree = Trie(companies_data['Company Name'].tolist())
    companies_similar_tree = PrefixTree(companies_data['Company Name'].tolist())
    return prefix_tree, companies_tree, companies_similar_tree

class Trie:
    def __init__(self, words):
        self.root = {}
        for word in words:
            current = self.root
            for letter in word:
                if letter not in current:
                    current[letter] = {}
                current = current[letter]
            current['$'] = True

class PrefixTree:
    def __init__(self, words):
        self.root = {}
        for word in words:
            current = self.root
            for letter in word:
                if letter not in current:
                    current[letter] = {}
                current = current[letter]
            current['$'] = True

class TestCompanyFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.opf_data = pd.DataFrame({
            'prefix': ['Товариство з обмеженою відповідальністю', 'Приватне підприємство', 'Селянське фермерське господарство'],
            'norm': ['тов', 'пп', 'сфг'],
            'count': [100, 50, 10]
        })
        cls.companies_data = pd.DataFrame({
            'Company Name': [
                'СЕЛЯНСЬКЕ (ФЕРМЕРСЬКЕ) ГОСПОДАРСТВО ГАЛАЄВА ВОЛОДИМИРА ПАВЛОВИЧА',
                'Таргет–Агро',
                '«ЕЛІТА СЕЛЕКТ»',
                '«ІНТЕГРАЛ СОЛЮШЕН УКРАЇНА»',
                'БЛАГОДІЙНИЙ ФОНД "" ЯНГОЛИ НЕБА""',
                '«ЛАРКСТОН ІНВЕСТ»',
                'АЛЬФАТЕХ ІНЖИНІРИНГ',
                'РІЧ-2022',
                'КАРДИНАЛ УНІВЕРСИТЕТ',
                'ОСТАПЕНКА МИКОЛИ МИКОЛАЙОВИЧА'
            ],
            'Company Number': [
                '31727690', '44677424', '44706072', '44736437', '44765101',
                '44904283', '44890035', '44922103', '45020116', '45036896'
            ]
        })
        cls.prefix_tree, cls.companies_tree, cls.companies_similar_tree = build_trees(cls.opf_data, cls.companies_data)

    def test_prepare_name(self):
        self.assertEqual(prepare_name('СЕЛЯНСЬКЕ (ФЕРМЕРСЬКЕ) ГОСПОДАРСТВО ГАЛАЄВА ВОЛОДИМИРА ПАВЛОВИЧА'), 
                         'селянське фермерське господарство галаєва володимира павловича')

    def test_normalize_name(self):
        self.assertEqual(normalize_name('СЕЛЯНСЬКЕ (ФЕРМЕРСЬКЕ) ГОСПОДАРСТВО ГАЛАЄВА ВОЛОДИМИРА ПАВЛОВИЧА', self.prefix_tree), 
                         'галаєва володимира павловича')

    def test_find_company_exact_match(self):
        result = find_company("СЕЛЯНСЬКЕ (ФЕРМЕРСЬКЕ) ГОСПОДАРСТВО ГАЛАЄВА ВОЛОДИМИРА ПАВЛОВИЧА", 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '31727690')

    def test_find_company_case_insensitive(self):
        result = find_company("селянське (фермерське) господарство галаєва володимира павловича", 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '31727690')

    def test_find_company_partial_match(self):
        result = find_company("ГАЛАЄВА ВОЛОДИМИРА ПАВЛОВИЧА", 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '31727690')

    def test_prepare_name_with_quotes(self):
        self.assertEqual(prepare_name('«ЕЛІТА СЕЛЕКТ»'), 'еліта селект')

    def test_normalize_name_with_prefix(self):
        self.assertEqual(normalize_name('ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ «ІНТЕГРАЛ СОЛЮШЕН УКРАЇНА»', self.prefix_tree), 
                         'інтеграл солюшен україна')

    def test_find_company_with_quotes(self):
        result = find_company('«ІНТЕГРАЛ СОЛЮШЕН УКРАЇНА»', 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '44736437')

    def test_prepare_name_with_multiple_spaces(self):
        self.assertEqual(prepare_name('БЛАГОДІЙНИЙ   ФОНД   ""  ЯНГОЛИ   НЕБА  ""'), 
                         'благодійний фонд янголи неба')

    def test_find_company_with_abbreviated_prefix(self):
        result = find_company('ТОВ «ЛАРКСТОН ІНВЕСТ»', 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '44904283')

    def test_normalize_name_with_abbreviated_prefix(self):
        self.assertEqual(normalize_name('ТОВ «ЛАРКСТОН ІНВЕСТ»', self.prefix_tree), 'ларкстон інвест')

    def test_find_company_with_typo(self):
        result = find_company("АЛЬФАТЕХ ІНЖІНІРИНГ", 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '44890035')

    def test_prepare_name_with_special_characters(self):
        self.assertEqual(prepare_name('РІЧ-2022'), 'річ 2022')

    def test_find_company_with_partial_name(self):
        result = find_company("КАРДИНАЛ", 
                              self.prefix_tree, self.companies_tree, self.companies_similar_tree, self.companies_data, 0.5)
        self.assertEqual(result[1], '45020116')

    def test_normalize_name_with_multiple_prefixes(self):
        self.assertEqual(normalize_name('СЕЛЯНСЬКЕ ФЕРМЕРСЬКЕ ГОСПОДАРСТВО ОСТАПЕНКА МИКОЛИ МИКОЛАЙОВИЧА', self.prefix_tree), 
                         'остапенка миколи миколайовича')

if __name__ == '__main__':
    unittest.main()
