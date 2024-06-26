import pandas as pd
from tqdm import tqdm
import re
import random
from functools import lru_cache
import Levenshtein

@lru_cache(maxsize=3000)
def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)

def normalized_levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return (max_len - levenshtein_distance(s1, s2)) / max_len

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
                find_prefix += " " + word
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
                        find_prefix += " " + best_match
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

    def search(self, word, max_errors):
        current_row = range(len(word) + 1)
        best_match = (None, float('inf'))

        for char in self.root.children:
            best_match = self._search_recursive(self.root.children[char], char, word, current_row, best_match, max_errors, char)

        return best_match

    def _search_recursive(self, node, char, word, previous_row, best_match, max_errors, current_prefix):
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

        if current_row[-1] <= max_errors and node.is_end_of_word:
            if current_row[-1] < best_match[1]:
                best_match = (current_prefix, current_row[-1])

        if min(current_row) <= max_errors:
            for next_char in node.children:
                best_match = self._search_recursive(
                    node.children[next_char], next_char, word, current_row, best_match, max_errors, current_prefix + next_char
                )

        return best_match

def load_data(opf_file_path, companies_file_path):
    opf_data = pd.read_csv(opf_file_path)
    opf_data = opf_data[opf_data['count'] > 5]
    all_prefixes = [val for val in opf_data.prefix.dropna().values] + [val for val in opf_data.norm.dropna().values]

    companies_data = pd.read_csv(companies_file_path)
    return all_prefixes, companies_data

def prepare_name(name):
    name = name.lower()
    chars_to_replace = "()\"«»'-"
    pattern = f"[{re.escape(chars_to_replace)}]"
    return re.sub(pattern, ' ', name).replace("  ", " ")

def normalize_name(name, prefix_tree):
    name = prepare_name(name)
    _, remaining_text, _, _ = prefix_tree.find_longest_prefix(name)
    return remaining_text.strip()  # This will keep the spaces between words

def build_trees(all_prefixes, companies_data):
    prefix_tree = PrefixTree()
    for name in tqdm(all_prefixes):
        prefix_tree.add_sequence(name)

    companies_data['normalized_name'] = companies_data['Company Name'].apply(lambda x: normalize_name(x, prefix_tree))

    companies_tree = PrefixTree()
    for name in tqdm(companies_data['normalized_name'].values):
        companies_tree.add_sequence(name.replace(" ", ""))

    companies_similar_tree = Trie()
    for name in tqdm(companies_data['normalized_name'].values):
        companies_similar_tree.add_sequence(name.replace(" ", ""))

    return prefix_tree, companies_tree, companies_similar_tree

def find_company(name, prefix_tree, companies_tree, companies_similar_tree, companies_data, min_similarity):
    opf, possible_name, prefix_similarity, find_prefix = prefix_tree.find_longest_prefix(name)
    if opf is None:
        opf, possible_name, prefix_similarity, find_prefix = prefix_tree.find_longest_prefix(name, allow_partial=True, min_similarity=min_similarity)
    possible_name = possible_name.strip()  # Remove leading/trailing spaces but keep internal spaces
    find_name, _, name_similarity, _ = companies_tree.find_longest_prefix(possible_name.replace(" ", ""))
    errors = 0
    company_id = None
    if find_name is None:
        find_name, errors = companies_similar_tree.search(possible_name.replace(" ", ""), 1)
    
    if find_name:
        # Find the corresponding company_id
        matching_company = companies_data[companies_data['normalized_name'].str.replace(" ", "") == find_name]
        if not matching_company.empty:
            company_id = matching_company.iloc[0]['Company Number']
    
    return opf, find_prefix, possible_name, find_name, prefix_similarity, errors, company_id

def process_query(query, prefix_tree, companies_tree, companies_similar_tree, companies_data):
    name, count, min_probability = query['name'], query['count'], query['min_probability']
    results = []
    opf, find_prefix, possible_name, find_name, prefix_similarity, errors, company_id = find_company(name, prefix_tree, companies_tree, companies_similar_tree, companies_data, min_probability)

    if find_name:
        probability = 1 - (errors / len(possible_name))  # Simplified probability calculation
        if probability >= min_probability:
            results.append({
                "name": find_name,
                "legal_id": company_id,
                "prefix_similarity": prefix_similarity,
                "probability": probability
            })

    return results

def main():
    opf_file_path = './data/companies_prefix_opf.csv'
    companies_file_path = './data/edr_ua_short.csv'
    all_prefixes, companies_data = load_data(opf_file_path, companies_file_path)

    prefix_tree, companies_tree, companies_similar_tree = build_trees(all_prefixes, companies_data)

    name = input("Enter the company name: ")
    count = int(input("Enter the count: "))
    min_probability = float(input("Enter the minimum probability: "))
    query = {"name": name, "count": count, "min_probability": min_probability}

    results = process_query(query, prefix_tree, companies_tree, companies_similar_tree, companies_data)
    print(results)

if __name__ == "__main__":
    main()
