import gzip
import json
import os
import os.path as osp
import pickle
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from ogb.utils.url import download_url
from tqdm import tqdm
from typing import Union

from stark_qa.skb.knowledge_base import SKB
from stark_qa.tools.download_hf import download_hf_file
from stark_qa.tools.io import load_files, save_files
from stark_qa.tools.node import Node, df_row_to_dict, register_node
from stark_qa.tools.process_text import clean_data, compact_text


DATASET = {
    "repo": "snap-stanford/stark",
    "processed": "skb/amazon/processed.zip",
    "metadata": "skb/amazon/category_list.json"
}

RAW_DATA_HEADER = {
    'review_header': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2',
    'qa_header': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon/qa'
}

class AmazonSKB(SKB):
    
    REVIEW_CATEGORIES = set([
        'Amazon_Fashion', 'All_Beauty', 'Appliances', 'Arts_Crafts_and_Sewing',
        'Automotive', 'Books', 'CDs_and_Vinyl', 'Cell_Phones_and_Accessories',
        'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Electronics', 'Gift_Cards',
        'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Industrial_and_Scientific',
        'Kindle_Store', 'Luxury_Beauty', 'Magazine_Subscriptions', 'Movies_and_TV',
        'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
        'Prime_Pantry', 'Software', 'Sports_and_Outdoors', 'Tools_and_Home_Improvement',
        'Toys_and_Games', 'Video_Games'
    ])
    
    QA_CATEGORIES = set([
        'Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 'Baby', 'Beauty',
        'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics',
        'Grocery_and_Gourmet_Food', 'Health_and_Personal_Care', 'Home_and_Kitchen',
        'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
        'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games'
    ])
    
    COMMON = set([
        'Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 'Cell_Phones_and_Accessories',
        'Clothing_Shoes_and_Jewelry', 'Electronics', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen',
        'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
        'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games'
    ])
    
    link_columns = ['also_buy', 'also_view']
    review_columns = [
        'reviewerID', 'summary', 'style', 'reviewText', 'vote', 'overall', 
        'verified', 'reviewTime'
    ]
    qa_columns = [
        'questionType', 'answerType', 'question', 'answer', 'answerTime'
    ]
    meta_columns = [
        'asin', 'title', 'global_category', 'category', 'price', 'brand', 
        'feature', 'rank', 'details', 'description'
    ]
    candidate_types = ['product']
    node_attr_dict = {
        'product': ['title', 'dimensions', 'weight', 'description', 'features', 'reviews', 'Q&A'],
        'brand': ['brand_name'],
        'category': ['category_name'],
        'color': ['color_name']
    }

    def __init__(self, 
                 root: Union[str, None] = None, 
                 categories: list = ['Sports_and_Outdoors'], 
                 meta_link_types: list = ['brand', 'category', 'color'],
                 max_entries: int = 25, 
                 download_processed: bool = True, 
                 **kwargs):
        """
        Initialize the AmazonSKB class.

        Args:
            root (Union[str, None]): Root directory to store the dataset. If None, default HF cache paths will be used.
            categories (list): Product categories.
            meta_link_types (list): A list of entries in node info that are used to construct meta links.
            max_entries (int): Maximum number of review & QA entries to show in the description.
            download_processed (bool): Whether to download the processed data.
        """
        self.root = root
        self.max_entries = max_entries

        if download_processed:
            if (self.root is None) or (
                self.root is not None
                and not osp.exists(osp.join(self.root, "category_list.json"))
            ):
                sub_category_path = osp.join(self.root, "category_list.json") if self.root is not None else None
                self.sub_category_path = download_hf_file(
                    DATASET["repo"],
                    DATASET["metadata"],
                    repo_type="dataset",
                    save_as_file=sub_category_path,
                )
        
            if (self.root is None) or (
                self.root is not None
                and meta_link_types is not None
                and not osp.exists(
                    osp.join(
                        self.root,
                        "processed",
                        "cache",
                        "-".join(meta_link_types),
                        "node_info.pkl",
                    )
                )
            ):
                processed_path = hf_hub_download(
                    DATASET["repo"], DATASET["processed"], repo_type="dataset"
                )
                if self.root is None:
                    self.root = osp.dirname(processed_path)
                if not osp.exists(
                    osp.join(
                        self.root,
                        "processed",
                        "cache",
                        "-".join(meta_link_types),
                        "node_info.pkl",
                    )):
                    with zipfile.ZipFile(processed_path, "r") as zip_ref:
                        zip_ref.extractall(path=self.root)
                    print(f"Extracting downloaded processed data to {self.root}")
                    

        self.raw_data_dir = osp.join(self.root, "raw")
        self.processed_data_dir = osp.join(osp.join(self.root, "processed"))
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        cache_path = None if meta_link_types is None else osp.join(self.processed_data_dir, 'cache', '-'.join(meta_link_types))
        

        if cache_path is not None and osp.exists(cache_path):
            print(f"Loading from {self.processed_data_dir}!")
            print(f'Loading cached graph with meta link types {meta_link_types}')
            processed_data = load_files(cache_path)
        else:
            print('Start processing raw data...')
            print(f'{meta_link_types=}')
            processed_data = self._process_raw(categories)
            if meta_link_types:
                processed_data = self.post_process(processed_data, meta_link_types=meta_link_types, cache_path=cache_path)
        
        super(AmazonSKB, self).__init__(**processed_data, **kwargs)
    
    def __getitem__(self, idx: int) -> Node:
        """
        Get the node at the specified index.

        Args:
            idx (int): Index of the node.

        Returns:
            Node: The node at the specified index.
        """
        idx = int(idx)
        node_info = self.node_info[idx]
        node = Node()
        register_node(node, node_info)
        return node
        
    def get_chunk_info(self, idx: int, 
                       attribute: str) -> str:
        """
        Get chunk information for the specified attribute.

        Args:
            idx (int): Index of the node.
            attribute (str): Attribute to get chunk information for.

        Returns:
            str: Chunk information.
        """
        if not hasattr(self[idx], attribute):
            return ''
        node_attr = getattr(self[idx], attribute)
        
        if 'feature' in attribute:
            features = [feature for feature in node_attr if feature and 'asin' not in feature.lower()]
            chunk = ' '.join(features)
        
        elif 'review' in attribute:
            chunk = ''
            if node_attr:
                scores = [0 if pd.isnull(review['vote']) else int(review['vote'].replace(",", "")) for review in node_attr]
                ranks = np.argsort(-np.array(scores))
                for idx, review_idx in enumerate(ranks):
                    review = node_attr[review_idx]
                    chunk += f'The review "{review["summary"]}" states that "{review["reviewText"]}". '
                    if idx > self.max_entries:
                        break
        
        elif 'qa' in attribute:
            chunk = ''
            if node_attr:
                for idx, question in enumerate(node_attr):
                    chunk += f'The question is "{question["question"]}", and the answer is "{question["answer"]}". '
                    if idx > self.max_entries:
                        break
        
        elif 'description' in attribute and node_attr:
            chunk = " ".join(node_attr)
    
        else:
            chunk = node_attr
        return chunk 
    
    def get_doc_info(self, idx: int, 
                     add_rel: bool = False, 
                     compact: bool = False) -> str:
        """
        Get document information for the specified node.

        Args:
            idx (int): Index of the node.
            add_rel (bool): Whether to add relationship information.
            compact (bool): Whether to compact the text.

        Returns:
            str: Document information.
        """
        if self.node_type_dict[int(self.node_types[idx])] == 'brand':
            return f'brand name: {self[idx].brand_name}'
        if self.node_type_dict[int(self.node_types[idx])] == 'category':
            return f'category name: {self[idx].category_name}'
        if self.node_type_dict[int(self.node_types[idx])] == 'color':
            return f'color name: {self[idx].color_name}'
        
        node = self[idx]
        doc = f'- product: {node.title}\n'
        if hasattr(node, 'brand'):
            doc += f'- brand: {node.brand}\n'
        try:
            dimensions, weight = node.details.dictionary.product_dimensions.split(' ; ')
            doc += f'- dimensions: {dimensions}\n- weight: {weight}\n'
        except:
            pass
        if node.description:
            description = " ".join(node.description).strip(" ")
            if description:
                doc += f'- description: {description}\n'
        
        feature_text = '- features: \n'
        if node.feature:
            for feature_idx, feature in enumerate(node.feature):
                if feature and 'asin' not in feature.lower():
                    feature_text += f'#{feature_idx + 1}: {feature}\n'
        else:
            feature_text = ''
        
        if node.review:
            review_text = '- reviews: \n'
            scores = [0 if pd.isnull(review['vote']) else int(review['vote'].replace(",", "")) for review in node.review]
            ranks = np.argsort(-np.array(scores))
            for i, review_idx in enumerate(ranks):
                review = node.review[review_idx]
                review_text += f'#{review_idx + 1}:\nsummary: {review["summary"]}\ntext: "{review["reviewText"]}"\n'
                if i > self.max_entries:
                    break
        else:
            review_text = ''
        
        if node.qa:
            qa_text = '- Q&A: \n'
            for qa_idx, qa in enumerate(node.qa):
                qa_text += f'#{qa_idx + 1}:\nquestion: "{qa["question"]}"\nanswer: "{qa["answer"]}"\n'
                if qa_idx > self.max_entries:
                    break
        else:
            qa_text = ''
        
        doc += feature_text + review_text + qa_text
        
        if add_rel:
            doc += self.get_rel_info(idx)
        if compact:
            doc = compact_text(doc)
        return doc
    
    def get_rel_info(self, 
                     idx: int, 
                     rel_types: Union[list, None] = None,
                     n_rel: int = -1) -> str:
        """
        Get relation information for the specified node.

        Args:
            idx (int): Index of the node.
            rel_types (Union[list, None]): List of relation types or None if all relation types are included.
            n_rel (int): Number of relations. Default is -1 if all relations are included.

        Returns:
            doc (str): Relation information.
        """
        doc = ''
        rel_types = self.rel_type_lst() if rel_types is None else rel_types
        n_also_buy = self.get_neighbor_nodes(idx, 'also_buy')
        n_also_view = self.get_neighbor_nodes(idx, 'also_view')
        n_has_brand = self.get_neighbor_nodes(idx, 'has_brand')

        str_also_buy = [f"#{idx + 1}: " + self[i].title + '\n' for idx, i in enumerate(n_also_buy)]
        str_also_view = [f"#{idx + 1}: " + self[i].title + '\n' for idx, i in enumerate(n_also_view)]
        if n_rel > 0:
            str_also_buy = str_also_buy[:n_rel]
            str_also_view = str_also_view[:n_rel]
        
        if not str_also_buy:
            str_also_buy = ''
        if not str_also_view:
            str_also_view = ''
        str_has_brand = ''
        if n_has_brand:
            str_has_brand = f'  brand: {self[n_has_brand[0]].brand_name}\n'
            
        str_also_buy = ''.join(str_also_buy)
        str_also_view = ''.join(str_also_view)

        if str_also_buy:
            doc += f'  products also purchased: \n{str_also_buy}'
        if str_also_view:
            doc += f'  products also viewed: \n{str_also_view}'
        if n_has_brand:
            doc += str_has_brand
            
        if doc:
            doc = '- relations:\n' + doc
        return doc
    
    def _process_raw(self, categories: list) -> dict:
        """
        Process raw data to construct the knowledge base.

        Args:
            categories (list): List of categories to process.

        Returns:
            dict: Processed data.
        """
        if 'all' in categories:
            review_categories = self.REVIEW_CATEGORIES
            qa_categories = self.QA_CATEGORIES
        else:
            qa_categories = review_categories = categories
            assert not set(categories) - self.COMMON, 'invalid categories exist'
        
        if osp.exists(osp.join(self.processed_data_dir, 'node_info.pkl')):
            print(f'Load processed data from {self.processed_data_dir}')
            loaded_files = load_files(self.processed_data_dir)
            loaded_files.update({
                'node_types': torch.zeros(len(loaded_files['node_info'])),
                'node_type_dict': {0: 'product'}
            })
            return loaded_files
        
        print('Check data downloading...')
        for category in review_categories:
            review_header = RAW_DATA_HEADER['review_header']
            if not os.path.exists(osp.join(self.raw_data_dir, f'{category}.json.gz')):
                print(f'Downloading {category} data...')
                download_url(f'{review_header}/categoryFiles/{category}.json.gz', self.raw_data_dir)
                download_url(f'{review_header}/metaFiles2/meta_{category}.json.gz', self.raw_data_dir)
        for category in qa_categories:
            qa_header = RAW_DATA_HEADER['qa_header']
            if not os.path.exists(osp.join(self.raw_data_dir, f'qa_{category}.json.gz')):
                print(f'Downloading {category} QA data...')
                download_url(f'{qa_header}/qa_{category}.json.gz', self.raw_data_dir)
            
        if not osp.exists(osp.join(self.processed_data_dir, 'node_info.pkl')):
            ckt_path = osp.join(self.root, 'intermediate')
            os.makedirs(ckt_path, exist_ok=True)
            print('Loading data... It might take a while')
            
            df_qa_path = os.path.join(ckt_path, 'df_qa.pkl')
            if os.path.exists(df_qa_path):
                df_qa = pd.read_pickle(df_qa_path)
            else:
                df_qa = pd.concat([
                    read_qa(osp.join(self.raw_data_dir, f'qa_{category}.json.gz'))
                    for category in qa_categories
                ])[['asin'] + self.qa_columns]
                df_qa.to_pickle(df_qa_path)
            print('df_qa loaded')
            
            df_review_path = os.path.join(ckt_path, 'df_review.pkl')
            if os.path.exists(df_review_path):
                df_review = pd.read_pickle(df_review_path)
            else:
                df_review = pd.concat([
                    read_review(osp.join(self.raw_data_dir, f'{category}.json.gz')) 
                    for category in review_categories
                ])[['asin'] + self.review_columns]
                df_review.to_pickle(df_review_path)
            print('df_review loaded')
            
            df_ucsd_meta_path = os.path.join(ckt_path, 'df_ucsd_meta.pkl')
            if os.path.exists(df_ucsd_meta_path):
                df_ucsd_meta = pd.read_pickle(df_ucsd_meta_path)
            else:
                meta_df_lst = []
                for category in review_categories:
                    cat_review = read_review(osp.join(self.raw_data_dir, f'meta_{category}.json.gz'))
                    cat_review.insert(0, 'global_category', category.replace('_', ' '))
                    meta_df_lst.append(cat_review)
                df_ucsd_meta = pd.concat(meta_df_lst)
                df_ucsd_meta.to_pickle(df_ucsd_meta_path)
            print('df_ucsd_meta loaded')
            
            print('Preprocessing data...')
            df_ucsd_meta = df_ucsd_meta.drop_duplicates(subset='asin', keep='first')
            df_meta = df_ucsd_meta[self.meta_columns + self.link_columns]
            
            df_review_meta = df_review.merge(df_meta, left_on='asin', right_on='asin')
            unique_asin = np.unique(np.array(df_review_meta['asin']))
            
            df_qa_reduced = df_qa[df_qa['asin'].isin(unique_asin)]
            df_review_reduced = df_review[df_review['asin'].isin(unique_asin)]
            df_meta_reduced = df_meta[df_meta['asin'].isin(unique_asin)].reset_index()
            
            def get_map(df):
                asin2id, id2asin = {}, {}
                for idx in range(len(df)):
                    asin2id[df['asin'][idx]] = idx
                    id2asin[idx] = df['asin'][idx]
                return asin2id, id2asin

            print('Construct node info and graph...')
            self.asin2id, self.id2asin = get_map(df_meta_reduced)
            node_info = self.construct_raw_node_info(df_meta_reduced, df_review_reduced, df_qa_reduced)
            edge_index, edge_types = self.create_raw_product_graph(df_meta_reduced, columns=self.link_columns)
            edge_type_dict = {0: 'also_buy', 1: 'also_view'}
            processed_data = {
                'node_info': node_info, 
                'edge_index': edge_index, 
                'edge_types': edge_types,
                'edge_type_dict': edge_type_dict
            }
            
            print(f'Saving to {self.processed_data_dir}...')
            save_files(save_path=self.processed_data_dir, **processed_data)

        processed_data.update({
            'node_types': torch.zeros(len(processed_data['node_info'])),
            'node_type_dict': {0: 'product'}
        })
        return processed_data
    
    def post_process(self, raw_info: dict, meta_link_types: list, cache_path: str = None) -> dict:
        """
        Post-process the raw information to add meta link types.

        Args:
            raw_info (dict): Raw information.
            meta_link_types (list): List of meta link types to add.
            cache_path (str): Path to cache the processed data.

        Returns:
            dict: Post-processed data.
        """
        print(f'Adding meta link types {meta_link_types}')
        node_info = raw_info['node_info']
        edge_type_dict = raw_info['edge_type_dict']
        node_type_dict = raw_info['node_type_dict']
        node_types = raw_info['node_types'].tolist()
        edge_index = raw_info['edge_index'].tolist()
        edge_types = raw_info['edge_types'].tolist()
        
        n_e_types, n_n_types = len(edge_type_dict), len(node_type_dict)
        for i, link_type in enumerate(meta_link_types):
            if link_type == 'brand':
                values = np.array([node_info_i[link_type] for node_info_i in node_info.values() 
                                   if link_type in node_info_i.keys()])
                indices = np.array([idx for idx, node_info_i in enumerate(node_info.values()) 
                                    if link_type in node_info_i.keys()])
            elif link_type in ['category', 'color']:
                value_list, indice_list = [], []
                for idx, node_info_i in enumerate(node_info.values()):
                    if link_type in node_info_i.keys():
                        value_list.extend(node_info_i[link_type])
                        indice_list.extend([idx for _ in range(len(node_info_i[link_type]))])
                values, indices = np.array(value_list), np.array(indice_list)
            else:
                raise Exception(f'Invalid meta link type {link_type}')
            
            cur_n_nodes = len(node_info)
            node_type_dict[n_n_types + i] = link_type
            edge_type_dict[n_e_types + i] = "has_" + link_type
            unique = np.unique(values)
            for j, unique_j in tqdm(enumerate(unique)):
                node_info[cur_n_nodes + j] = {link_type + '_name': unique_j}
                ids = indices[np.array(values == unique_j)]
                edge_index[0].extend(ids.tolist())
                edge_index[1].extend([cur_n_nodes + j for _ in range(len(ids))])
                edge_types.extend([i + n_e_types for _ in range(len(ids))])
            node_types.extend([n_n_types + i for _ in range(len(unique))])
            print(f'finished adding {link_type}')
            
        edge_index = torch.LongTensor(edge_index)
        edge_types = torch.LongTensor(edge_types)
        node_types = torch.LongTensor(node_types)
        files = {
            'node_info': node_info, 
            'edge_index': edge_index, 
            'edge_types': edge_types, 
            'edge_type_dict': edge_type_dict,
            'node_type_dict': node_type_dict,
            'node_types': node_types
        }
        if cache_path is not None:
            save_files(cache_path, **files)
        return files
    
    def _process_brand(self, brand: str) -> str:
        """
        Process brand names to remove unnecessary characters.

        Args:
            brand (str): Brand name.

        Returns:
            str: Processed brand name.
        """
        brand = brand.strip(" \".*+,-_!@#$%^&*();\/|<>\'\t\n\r\\")
        if brand.startswith('by '):
            brand = brand[3:]
        if brand.endswith('.com'):
            brand = brand[:-4]
        if brand.startswith('www.'):
            brand = brand[4:]
        if len(brand) > 100:
            brand = brand.split(' ')[0]
        return brand
    
    def construct_raw_node_info(self, df_meta: pd.DataFrame, df_review: pd.DataFrame, df_qa: pd.DataFrame) -> dict:
        """
        Construct raw node information.

        Args:
            df_meta (pd.DataFrame): DataFrame containing meta information.
            df_review (pd.DataFrame): DataFrame containing review information.
            df_qa (pd.DataFrame): DataFrame containing QA information.

        Returns:
            dict: Dictionary containing node information.
        """
        node_info = {idx: {'review': [], 'qa': []} for idx in range(len(df_meta))}
        
        ###################### Assign color ########################
        def assign_colors(df_review, lower_limit=20):
            # asign to color
            df_review = df_review[['asin', 'style']]
            df_review = df_review.dropna(subset=['style'])
            raw_color_dict = {}
            for idx, row in tqdm(df_review.iterrows()):
                asin, style = row['asin'], row['style']
                for key in style.keys():
                    if 'color' in key.lower():
                        try:
                            raw_color_dict[asin] 
                        except:
                            raw_color_dict[asin] = []
                        raw_color_dict[asin].append(
                            style[key].strip().lower() if isinstance(style[key], str) else style[key][0].strip())
            
            all_color_values = []
            for asin in raw_color_dict.keys():
                raw_color_dict[asin] = list(set(raw_color_dict[asin]))
                all_color_values.extend(raw_color_dict[asin])
            
            print('number of all colors', len(all_color_values))
            color_counter = Counter(all_color_values)
            print('number of unique colors', len(color_counter))
            color_counter = {k: v for k, v in sorted(color_counter.items(), key=lambda item: item[1], reverse=True)}
            selected_colors = []
            for color, number in color_counter.items():
                if number > lower_limit and len(color) > 2 and len(color.split(' ')) < 5 and color.isnumeric() is False:
                    selected_colors.append(color)
            print('number of selected colors', len(selected_colors))
            
            filtered_color_dict = {}
            total_color_connections = 0
            for asin in raw_color_dict.keys():
                filtered_color_dict[asin] = []
                for value in raw_color_dict[asin]:
                    if value in selected_colors:
                        filtered_color_dict[asin].append(value)
                total_color_connections += len(filtered_color_dict[asin])
            print('number of linked products', len(filtered_color_dict))
            print('number of total connections', total_color_connections)
            return filtered_color_dict
    
        filtered_color_dict_path = os.path.join(self.root, 'intermediate', 'filtered_color_dict.pkl')
        if os.path.exists(filtered_color_dict_path):
            with open(filtered_color_dict_path, 'rb') as f:
                filtered_color_dict = pickle.load(f)
        else:
            filtered_color_dict = assign_colors(df_review)
            with open(filtered_color_dict_path, 'wb') as f:
                pickle.dump(filtered_color_dict, f)
        
        for df_meta_i in tqdm(df_meta.itertuples()):
            asin = df_meta_i.asin
            idx = self.asin2id[asin]
            if asin in filtered_color_dict and filtered_color_dict[asin]:
                node_info[idx]['color'] = filtered_color_dict[asin]
        
        ###################### Assign brand and category ########################
        sub_categories = set(json.load(open(self.sub_category_path, 'r')))
        for df_meta_i in tqdm(df_meta.itertuples()):
            asin = df_meta_i.asin
            idx = self.asin2id[asin]
            for column in self.meta_columns:
                if column == 'brand':
                    brand = self._process_brand(clean_data(getattr(df_meta_i, column)))
                    if brand:
                        node_info[idx]['brand'] = brand
                elif column == 'category':
                    category_list = [
                        category.lower() for category in getattr(df_meta_i, column)
                        if category.lower() in sub_categories
                    ]
                    if category_list:
                        node_info[idx]['category'] = category_list
                else:
                    node_info[idx][column] = clean_data(getattr(df_meta_i, column))
        
        ###################### Process review and QA ########################
        for name, df, colunm_names in zip(['review', 'qa'], 
                                          [df_review, df_qa], 
                                          [self.review_columns, self.qa_columns]):
            for i in tqdm(range(len(df))):
                df_i = df.iloc[i]
                asin = df_i['asin']
                idx = self.asin2id[asin]
                node_info[idx][name].append(df_row_to_dict(df_i, colunm_names))
        
        return node_info

    def create_raw_product_graph(self, df: pd.DataFrame, columns: list) -> tuple:
        """
        Create raw product graph.

        Args:
            df (pd.DataFrame): DataFrame containing meta information.
            columns (list): List of columns to create edges.

        Returns:
            tuple: Tuple containing edge index and edge types.
        """
        edge_types = []
        edge_index = [[], []]
        for df_i in df.itertuples():
            out_node = self.asin2id[df_i.asin]
            for edge_type_id, edge_type in enumerate(columns):
                if isinstance(getattr(df_i, edge_type), list):
                    in_nodes = [self.asin2id[i] for i in getattr(df_i, edge_type) if i in self.asin2id]
                    edge_types.extend([edge_type_id] * len(in_nodes))
                    edge_index[0].extend([out_node] * len(in_nodes))
                    edge_index[1].extend(in_nodes)
        return torch.LongTensor(edge_index), torch.LongTensor(edge_types)

    def has_brand(self, idx: int, brand: str) -> bool:
        """
        Check if the node has the specified brand.

        Args:
            idx (int): Index of the node.
            brand (str): Brand name.

        Returns:
            bool: Whether the node has the specified brand.
        """
        try: 
            b = self[idx].brand
            if b.endswith('.com'):
                b = b[:-4]
            if brand.endswith('.com'):
                brand = brand[:-4]
            return b.lower().strip("\"") == brand.lower().strip("\"")
        except:
            return False

    def has_also_buy(self, idx: int, also_buy_item: int) -> bool:
        """
        Check if the node has the specified also_buy item.

        Args:
            idx (int): Index of the node.
            also_buy_item (int): Item to check.

        Returns:
            bool: Whether the node has the specified also_buy item.
        """
        try: 
            also_buy_lst = self.get_neighbor_nodes(idx, 'also_buy') 
            return also_buy_item in also_buy_lst
        except:
            return False
        
    def has_also_view(self, idx: int, also_view_item: int) -> bool:
        """
        Check if the node has the specified also_view item.

        Args:
            idx (int): Index of the node.
            also_view_item (int): Item to check.

        Returns:
            bool: Whether the node has the specified also_view item.
        """
        try: 
            also_view_lst = self.get_neighbor_nodes(idx, 'also_view') 
            return also_view_item in also_view_lst
        except:
            return False


def read_review(path: str) -> pd.DataFrame:
    """
    Read and parse review files.

    Args:
        path (str): Path to the review file.

    Returns:
        pd.DataFrame: DataFrame containing the reviews.
    """
    def parse(path: str):
        with gzip.open(path, 'rb') as g:
            for l in g:
                yield json.loads(l)

    def getDF(path: str) -> pd.DataFrame:
        df = {}
        for i, d in enumerate(parse(path)):
            df[i] = d
        return pd.DataFrame.from_dict(df, orient='index')

    return getDF(path)


def read_qa(path: str) -> pd.DataFrame:
    """
    Read and parse QA files.

    Args:
        path (str): Path to the QA file.

    Returns:
        pd.DataFrame: DataFrame containing the QA data.
    """
    def parse(path: str):
        with gzip.open(path, 'rb') as g:
            for l in g:
                yield eval(l)

    def getDF(path: str) -> pd.DataFrame:
        df = {}
        for i, d in enumerate(parse(path)):
            df[i] = d
        return pd.DataFrame.from_dict(df, orient='index')

    return getDF(path)
