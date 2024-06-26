import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrefixTree:
    def __init__(self):
        self.root = {}

    def add_sequence(self, sequence):
        node = self.root
        for word in sequence.lower().split():
            if word not in node:
                node[word] = {}
            node = node[word]
        node[None] = None  # маркер конца последовательности

    def find_longest_prefix(self, text, allow_partial=False, min_similarity=0.5):
        words = text.lower().split()
        node = self.root
        matched_prefix_length = 0
        matched_prefix_node = None
        total_similarity = 1.0
        find_prefix = ''

        for i, word in enumerate(words):
            if word in node:
                node = node[word]
                find_prefix = find_prefix + " " + word
                if None in node:  # найден конец последовательности
                    matched_prefix_length = i + 1
                    matched_prefix_node = node
            else:
                if allow_partial:
                    best_match = None
                    best_similarity = 0
                    for child_word in node.keys():
                        if child_word is not None:
                            similarity = normalized_levenshtein_similarity(word, child_word)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = child_word
                    if best_similarity >= min_similarity:
                        node = node[best_match]
                        find_prefix = find_prefix + " " + best_match
                        total_similarity *= best_similarity
                        if None in node:  # найден конец последовательности
                            matched_prefix_length = i + 1
                            matched_prefix_node = node
                    else:
                        break
                else:
                    break

        if matched_prefix_node and None in matched_prefix_node:
            remaining_text = ' '.join(words[matched_prefix_length:])
            matched_prefix = ' '.join(words[:matched_prefix_length])
            return matched_prefix, remaining_text, total_similarity, find_prefix.strip()
        else:
            return None, text, 0.0, None  # не найдено совпадение

    def print_tree(self, depth=3):
        self._print_node(self.root, depth, 0)

    def _print_node(self, node, max_depth, current_depth):
        if current_depth > max_depth:
            return
        for word, child in node.items():
            if word is None:
                print(' ' * current_depth * 4 + '<END>')
            else:
                print(' ' * current_depth * 4 + word)
                self._print_node(child, max_depth, current_depth + 1)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add_sequence(self, sequence):
        node = self.root
        for char in sequence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word, max_errors, trace=False):
        current_row = range(len(word) + 1)
        best_match = (None, float('inf'))

        if trace:
            print(f"Searching for '{word}' with max_errors={max_errors}")

        for char in self.root.children:
            best_match = self._search_recursive(self.root.children[char], char, word, current_row, best_match, max_errors, char, trace)

        return best_match

    def _search_recursive(self, node, char, word, previous_row, best_match, max_errors, current_prefix, trace):
        columns = len(word) + 1
        current_row = [previous_row[0] + 1]

        for column in range(1, columns):
            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word[column - 1] != char:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        if trace:
            print(f"Prefix '{current_prefix}': current_row={current_row}, previous_row={previous_row}")

        if current_row[-1] <= max_errors and node.is_end_of_word:
            if current_row[-1] < best_match[1]:
                best_match = (current_prefix, current_row[-1])
                if trace:
                    print(f"New best match: {best_match}")

        if min(current_row) <= max_errors:
            for next_char in node.children:
                best_match = self._search_recursive(
                    node.children[next_char], next_char, word, current_row, best_match, max_errors, current_prefix + next_char, trace
                )

        return best_match

def load_data(opf_file_path: str, companies_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading data...")
    opf_data = pd.read_csv(opf_file_path)
    opf_data = opf_data[opf_data['count'] > 5]
    companies_data = pd.read_csv(companies_file_path)
    return opf_data, companies_data

def prepare_prefix_tree(opf_data: pd.DataFrame) -> PrefixTree:
    logging.info("Preparing prefix tree...")
    all_prefixes = [val for val in opf_data.prefix.dropna().values] + [val for val in opf_data.norm.dropna().values]
    prefix_tree = PrefixTree()
    for name in tqdm(all_prefixes):
        prefix_tree.add_sequence(name)
    return prefix_tree

def prepare_name(name: str) -> str:
    name = name.lower()
    chars_to_replace = "()\"«»'-"
    pattern = f"[{re.escape(chars_to_replace)}]"
    return re.sub(pattern, ' ', name).replace("  ", " ")

def normalize_name(name: str, prefix_tree: PrefixTree) -> str:
    name = prepare_name(name)
    _, remaining_text, _, _ = prefix_tree.find_longest_prefix(name)
    return remaining_text.strip()

def prepare_company_trees(companies_data: pd.DataFrame) -> Tuple[PrefixTree, Trie]:
    logging.info("Preparing company trees...")
    companies_tree = PrefixTree()
    companies_similar_tree = Trie()
    for name in tqdm(companies_data['normalized_name'].values):
        companies_tree.add_sequence(name.replace(" ", ""))
        companies_similar_tree.add_sequence(name.replace(" ", ""))
    return companies_tree, companies_similar_tree

def find_company(name: str, prefix_tree: PrefixTree, companies_tree: PrefixTree, 
                 companies_similar_tree: Trie, companies_data: pd.DataFrame) -> Tuple:
    opf, possible_name, prefix_similarity, find_prefix = prefix_tree.find_longest_prefix(name)
    if opf is None:
        opf, possible_name, prefix_similarity, find_prefix = prefix_tree.find_longest_prefix(name, allow_partial=True, min_similarity=0.7)
    possible_name = possible_name.strip()
    find_name, _, name_similarity, _ = companies_tree.find_longest_prefix(possible_name.replace(" ", ""))
    errors = 0
    company_id = None
    if find_name is None:
        find_name, errors = companies_similar_tree.search(possible_name.replace(" ", ""), 1)
    
    if find_name:
        matching_company = companies_data[companies_data['normalized_name'].str.replace(" ", "") == find_name]
        if not matching_company.empty:
            company_id = matching_company.iloc[0]['Company Number']
    
    return opf, find_prefix, possible_name, find_name, prefix_similarity, errors, company_id

def main():
    try:
        opf_file_path = './companies_prefix_opf.csv'
        companies_file_path = './edr_ua_short.csv'

        opf_data, companies_data = load_data(opf_file_path, companies_file_path)
        prefix_tree = prepare_prefix_tree(opf_data)

        companies_data['normalized_name'] = companies_data['Company Name'].apply(lambda x: normalize_name(x, prefix_tree))

        companies_tree, companies_similar_tree = prepare_company_trees(companies_data)

        # Example usage
        test_company = "ТОВ Агро техніка"
        result = find_company(test_company, prefix_tree, companies_tree, companies_similar_tree, companies_data)
        logging.info(f"Result for '{test_company}': {result}")

        # Additional processing or batch operations can be added here

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()