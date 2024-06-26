import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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