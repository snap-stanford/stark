import os
import os.path as osp
import gzip
import pickle
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ogb.utils.url import download_url
from src.benchmarks.semistruct.knowledge_base import SemiStructureKB
from src.tools.process_text import clean_data, compact_text
from src.tools.node import df_row_to_dict, Node, register_node
from src.tools.io import save_files, load_files


class AmazonSemiStruct(SemiStructureKB):
    
    REVIEW_CATEGORIES = set(['Amazon_Fashion','All_Beauty','Appliances',
                             'Arts_Crafts_and_Sewing','Automotive','Books',
                             'CDs_and_Vinyl','Cell_Phones_and_Accessories',
                             'Clothing_Shoes_and_Jewelry','Digital_Music',
                             'Electronics','Gift_Cards','Grocery_and_Gourmet_Food',
                             'Home_and_Kitchen','Industrial_and_Scientific', 'Kindle_Store',
                             'Luxury_Beauty','Magazine_Subscriptions', 'Movies_and_TV',
                             'Musical_Instruments', 'Office_Products','Patio_Lawn_and_Garden',
                             'Pet_Supplies','Prime_Pantry','Software','Sports_and_Outdoors',
                             'Tools_and_Home_Improvement','Toys_and_Games','Video_Games'])
    
    # single answers
    QA_CATEGORIES = set(['Appliances','Arts_Crafts_and_Sewing', 'Automotive',
                         'Baby','Beauty','Cell_Phones_and_Accessories',
                         'Clothing_Shoes_and_Jewelry','Electronics',
                        'Grocery_and_Gourmet_Food','Health_and_Personal_Care',
                        'Home_and_Kitchen','Musical_Instruments','Office_Products',
                        'Patio_Lawn_and_Garden','Pet_Supplies','Sports_and_Outdoors',
                        'Tools_and_Home_Improvement','Toys_and_Games','Video_Games'])
    
    COMMON = set(['Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 
                  'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 
                  'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Musical_Instruments', 
                  'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Sports_and_Outdoors', 
                  'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games'])
    
    link_columns = ['also_buy', 'also_view']
    review_columns = ['reviewerID', 'summary', 'reviewText', 'vote', 'overall', 'verified', 'reviewTime']
    qa_columns = ['questionType', 'answerType', 'question', 'answer', 'answerTime']
    meta_columns = ['asin', 'title', 'global_category', 'category', 'price', 'brand', 'feature',
                    'rank', 'details', 'description']
    candidate_types = ['product']
    node_attr_dict = {'product': ['title', 'dimensions', 'weight', 'description', 'features', 'reviews', 'Q&A'],
                       'brand': ['brand_name']}
    
    def __init__(self, 
                 categories: list, 
                 path_qa=None,
                 path_review=None,
                 raw_data_dir=None, 
                 save_path=None, 
                 meta_link_types=None,
                 max_entries=25,
                 indirected=True):
        if path_qa is None or path_review is None:
            path_review = path_qa = raw_data_dir
        '''
            Args: 
                categories (list): product categories
                raw_data_dir (Path or str): Amazon QA data dir and review data dir
                save_path (Path or str): dir that stores processed data
                meta_link_types (list): a list which may contain entries in node info 
                                        that used to consruct meta links, e.g. ['category', 'brand'] 
                                        will construct entity nodes of catrgory and brand which link 
                                        to corresponding nodes
        '''
        self.max_entries = max_entries 
        # construct the graph based on link info in the raw data
        cache_path = None if (save_path is None or meta_link_types is None) else \
            osp.join(save_path, 'cache', '-'.join(meta_link_types))
        if not (cache_path is None) and osp.exists(cache_path):
            print(f'Load cached graph with meta link types {meta_link_types}')
            processed_data = load_files(cache_path)
        else:
            processed_data = self.process_raw(categories, path_qa, path_review, save_path)
            if meta_link_types: # custumize the graph by adding meta links
                processed_data = self.post_process(processed_data, meta_link_types=meta_link_types, cache_path=cache_path)
        super(AmazonSemiStruct, self).__init__(**processed_data, indirected=indirected)
    
    def __getitem__(self, idx):
        idx = int(idx)
        node_info = self.node_info[idx]
        try:
            dimensions, weight = node.details.dictionary.product_dimensions.split(' ; ')
            node_info['dimensions'], node_info['weight'] = dimensions, weight
        except: pass
        node = Node()
        register_node(node, node_info)
        return node
        
    def get_chunk_info(self, idx, attribute):
        if not hasattr(self[idx], attribute): return ''
        node_attr = getattr(self[idx], attribute)
        
        if 'feature' in attribute:
            features = []
            if len(node_attr):
                for feature_idx, feature in enumerate(node_attr):
                    if feature == '': continue
                    if 'asin' in feature.lower(): continue
                    features.append(feature)
            chunk = ' '.join(features)
        
        elif 'review' in attribute:
            chunk = ''
            if len(node_attr):
                scores = [0 if pd.isnull(review['vote']) else int(review['vote'].replace(",","")) for review in node_attr]
                ranks = np.argsort(-np.array(scores))
                for idx, review_idx in enumerate(ranks):
                    review = node_attr[review_idx]
                    chunk += 'The review \"' + str(review['summary']) + '\"'
                    chunk += 'states that \"' + str(review['reviewText']) + '\". '
                    if idx > self.max_entries: break
        
        elif 'qa' in attribute:
            chunk = ''
            if len(node_attr):
                for idx, question in enumerate(node_attr):
                    chunk += 'The question is \"' + str(question['question']) + '\", '
                    chunk += 'and the answer is \"' + str(question['answer']) + '\". '
                    if idx > self.max_entries: 
                        break
        
        elif 'description' in attribute and len(node_attr):
            chunk = " ".join(node_attr)
    
        else:
            chunk = node_attr
        return chunk 
    
    def get_doc_info(self, idx, 
                     add_rel=True, 
                     compact=False):
        
        if self.node_type_dict[int(self.node_types[idx])] == 'brand':
            return f'brand name: {self[idx].brand_name}'
        
        node = self[idx]
        doc = f'- product: {node.title}\n'
        if hasattr(node, 'brand'):
            doc += f'- brand: {node.brand}\n'
        try:
            dimensions, weight = node.details.dictionary.product_dimensions.split(' ; ')
            doc += (f'- dimensions: {dimensions}\n'
                    f'- weight: {weight}\n')
        except: pass
        if len(node.description):
            description = " ".join(node.description).strip(" ")
            if len(description) > 0:
                doc += f'- description: {description}\n'
        
        feature_text = f'- features: \n'
        if len(node.feature):
            for feature_idx, feature in enumerate(node.feature):
                if feature == '': continue
                if 'asin' in feature.lower(): continue
                feature_text += (f'#{feature_idx + 1}: {feature}\n')
        else: feature_text = ''
        
        if len(node.review):
            review_text = f'- reviews: \n'
            scores = [0 if pd.isnull(review['vote']) else int(review['vote'].replace(",","")) for review in node.review]
            ranks = np.argsort(-np.array(scores))
            for i, review_idx in enumerate(ranks):
                review = node.review[review_idx]
                review_text += (f'#{review_idx + 1}:\n'
                                f'summary: {review["summary"]}\n'
                                f'text: "{review["reviewText"]}"\n')
                if i > self.max_entries: break
        else: review_text = ''
        
        if len(node.qa):
            qa_text = f'- Q&A: \n'
            for qa_idx, qa in enumerate(node.qa):
                qa_text += (f'#{qa_idx + 1}:\n'
                            f'question: "{qa["question"]}"\n'
                            f'answer: "{qa["answer"]}"\n')
                if qa_idx > self.max_entries: break
        else: qa_text = ''
        
        doc += feature_text + review_text + qa_text
        
        if add_rel:
            doc += self.get_rel_info(idx)
        if compact: 
            doc = compact_text(doc)
        return doc
    
    def get_rel_info(self, idx, rel_types=None, n_rel=-1):
        doc = ''
        rel_types = self.rel_type_lst() if rel_types is None else rel_types
        n_also_buy = self.get_neighbor_nodes(idx, 'also_buy')
        n_also_view = self.get_neighbor_nodes(idx, 'also_view')
        n_has_brand = self.get_neighbor_nodes(idx, 'has_brand')

        str_also_buy = [f"#{idx + 1}: " + self[i].title + '\n' for idx, i in enumerate(n_also_buy)]
        str_also_view = [f"#{idx + 1}: " + self[i].title  + '\n' for idx, i in enumerate(n_also_view)]
        
        if len(str_also_buy) == 0: str_also_buy = ''
        if len(str_also_view) == 0: str_also_view = ''
        str_has_brand = ''
        if len(n_has_brand): 
            str_has_brand = f'  brand: {self[n_has_brand[0]].brand_name}\n'
            
        str_also_buy = ''.join(str_also_buy)
        str_also_view = ''.join(str_also_view)

        if len(str_also_buy):
            doc += f'  products also purchased: \n{str_also_buy}'
        if len(str_also_view):
            doc += f'  products also viewed: \n{str_also_view}'
        if len(n_has_brand):
            doc += str_has_brand
            
        if len(doc): 
            doc = '- relations:\n' + doc
        return doc
    
    def process_raw(self, categories, path_qa, path_review, save_path=None):
        if 'all' in categories:
            review_categories = self.REVIEW_CATEGORIES
            qa_categories = self.QA_CATEGORIES
        else:
            qa_categories = review_categories = categories
            assert len(set(categories) - self.COMMON) == 0, f'invalid categories exist'
        
        if not save_path is None and osp.exists(osp.join(save_path, 'node_info.pkl')):
            print(f'Load processed data from {save_path}')
            loaded_files = load_files(save_path)
            loaded_files.update(
                {'node_types': torch.zeros(len(loaded_files['node_info'])),
                 'node_type_dict': {0: 'product'}})
            return loaded_files
        
        print(f'Check data downloading...')
        for category in review_categories:
            review_header = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2'
            download_url(f'{review_header}/categoryFiles/{category}.json.gz', path_review)
            download_url(f'{review_header}/metaFiles2/meta_{category}.json.gz', path_review)
        for category in qa_categories:
            qa_header = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon/qa'
            download_url(f'{qa_header}/qa_{category}.json.gz', path_qa)
            
        if save_path is None or not osp.exists(osp.join(save_path, 'node_info.pkl')):
            print('Loading data... It might take a while')
            # read amazon QA data
            df_qa = pd.concat([read_qa(osp.join(path_qa, f'qa_{category}.json.gz'))
                               for category in qa_categories])[['asin'] + self.qa_columns]
            
            # read amazon review data
            df_review = pd.concat([read_review(osp.join(path_review, f'{category}.json.gz')) 
                                   for category in review_categories])[['asin'] + self.review_columns]
            # read amazon meta data from amazon review & amazon kdd
            meta_df_lst = []
            for category in review_categories:
                cat_review = read_review(osp.join(path_review, f'meta_{category}.json.gz'))
                cat_review.insert(0, 'global_category', category.replace('_', ' '))
                meta_df_lst.append(cat_review)
            df_ucsd_meta = pd.concat(meta_df_lst)
            
            # print('Cleaning data...')
            # cleaning QA and review data
            # for df in [df_qa, df_review]:
            #     for column in df.columns:
            #         df[column].replace('', np.nan, inplace=True)
            #     df.dropna(inplace=True)
            
            print('Preprocessing data...')
            df_ucsd_meta = df_ucsd_meta.drop_duplicates(subset='asin', keep='first')
            df_meta = df_ucsd_meta[self.meta_columns + self.link_columns]
            
            # Merge dataframes
            # df_qa_meta = df_qa.merge(df_meta, left_on='asin', right_on='asin')
            df_review_meta = df_review.merge(df_meta, left_on='asin', right_on='asin')
            unique_asin = np.unique(np.array(df_review_meta['asin']))
            # unique_asin = np.array(list(set(df_review_meta['asin']).intersection(set(df_qa_meta['asin']))))
            
            # Filer items with both meta and review data
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
            # get mapping from asin to node id and its reversed mapping
            self.asin2id, self.id2asin = get_map(df_meta_reduced)
            node_info = self.construct_raw_node_info(df_meta_reduced, df_review_reduced, df_qa_reduced)
            edge_index, edge_types = self.create_raw_product_graph(df_meta_reduced, 
                                                                   columns=self.link_columns)
            edge_type_dict = {0: 'also_buy', 1: 'also_view'}
            processed_data = {
                'node_info': node_info, 
                'edge_index': edge_index, 
                'edge_types': edge_types,
                'edge_type_dict': edge_type_dict}
            
            if save_path is not None:
                print(f'Saving to {save_path}...')
                save_files(save_path=save_path, **processed_data)

        processed_data.update({'node_types': torch.zeros(len(processed_data['node_info'])),
                               'node_type_dict': {0: 'product'}})
        return processed_data
    
    def post_process(self, raw_info, meta_link_types, cache_path=None):
        print(f'Adding meta link types {meta_link_types}')
        node_info = raw_info['node_info']
        edge_type_dict = raw_info['edge_type_dict']
        node_type_dict = raw_info['node_type_dict']
        node_types = raw_info['node_types'].tolist()
        edge_index = raw_info['edge_index'].tolist()
        edge_types = raw_info['edge_types'].tolist()
        
        n_e_types, n_n_types = len(edge_type_dict), len(node_type_dict)
        for i, link_type in enumerate(meta_link_types):
            values = np.array([self._process_brand(node_info_i[link_type]) for node_info_i in node_info.values() if link_type in node_info_i.keys()])
            indices = np.array([idx for idx, node_info_i in enumerate(node_info.values()) if link_type in node_info_i.keys()])
            
            cur_n_nodes = len(node_info)
            node_type_dict[n_n_types + i] = link_type
            edge_type_dict[n_e_types + i] = "has_" + link_type
            unique = np.unique(values)
            for j, unique_j in enumerate(unique):
                node_info[cur_n_nodes + j] = {link_type + '_name': unique_j}
                ids = indices[np.array(values == unique_j)]
                edge_index[0].extend(list(ids))
                edge_index[1].extend([cur_n_nodes + j for _ in range(len(ids))])
                edge_types.extend([i + n_e_types for _ in range(len(ids))])
            node_types.extend([n_n_types + i for _ in range(len(unique))])
        edge_index = torch.LongTensor(edge_index)
        edge_types = torch.LongTensor(edge_types)
        node_types = torch.LongTensor(node_types)
        files = {'node_info': node_info, 
                 'edge_index': edge_index, 
                 'edge_types': edge_types, 
                 'edge_type_dict': edge_type_dict,
                 'node_type_dict': node_type_dict,
                 'node_types': node_types
                 }
        if cache_path is not None:
            save_files(cache_path, **files)
        return files
    
    def _process_brand(self, brand):
        brand = brand.strip(" \".*+,-_!@#$%^&*();\/|<>\'\t\n\r\\")
        if len(brand) > 3 and brand[:3] == 'by ':
            brand = brand[3:]
        if len(brand) > 4 and brand[-4:] == '.com':
            brand = brand[:-4]
        if len(brand) > 4 and brand[:4] == 'www.':
            brand = brand[4:]
        if len(brand) > 100: 
            brand = brand.split(' ')[0]
        return brand
    
    def construct_raw_node_info(self, df_meta, df_review, df_qa):
        node_info = {}
        for idx, asin in self.id2asin.items():
            node_info[idx] = {}
            node_info[idx]['review'] = []
            node_info[idx]['qa'] = []
        
        for i in tqdm(range(len(df_meta))):
            df_meta_i = df_meta.iloc[i]
            asin = df_meta_i['asin']
            idx = self.asin2id[asin]
            for column in self.meta_columns:
                if column == 'brand':
                    brand = self._process_brand(clean_data(df_meta_i[column]))
                    if len(brand) > 1:
                        node_info[idx]['brand'] = brand
                else:
                    node_info[idx][column] = clean_data(df_meta_i[column])
                        
        for name, df in zip(['review', 'qa'], [df_review, df_qa]):
            for i in tqdm(range(len(df))):
                df_i = df.iloc[i]
                asin = df_i['asin']
                idx = self.asin2id[asin]
                node_info[idx][name].append(
                    df_row_to_dict(df_i, colunm_names=self.review_columns \
                                   if name == 'review' else self.qa_columns))
        return node_info

    def create_raw_product_graph(self, df, columns):
        edge_types = []
        edge_index = [[], []]
        for idx in range(len(df)):
            out_node = self.asin2id[df['asin'].iloc[idx]]
            for edge_type_id, edge_type in enumerate(columns):
                in_nodes = []
                if not isinstance(df[edge_type].iloc[idx], list):
                    continue
                for i in df[edge_type].iloc[idx]:
                    try:
                        in_nodes.append(self.asin2id[i])
                    except KeyError:
                        continue
                edge_types.extend([edge_type_id for _ in range(len(in_nodes))])
                edge_index[0].extend([out_node for _ in range(len(in_nodes))])
                edge_index[1].extend(in_nodes)
        return torch.LongTensor(edge_index), torch.LongTensor(edge_types)

    def has_brand(self, idx, brand):
        try: 
            b = self[idx].brand
            if len(b) > 4 and b[-4:] == '.com': b = b[:-4]
            if len(brand) > 4 and brand[-4:] == '.com': brand = brand[:-4]
            return b.lower().strip("\"") == brand.lower().strip("\"")
        except:
            return False

    def has_also_buy(self, idx, also_buy_item):
        try: 
            also_buy_lst = self.get_neighbor_nodes(idx, 'also_buy') 
            return also_buy_item in also_buy_lst
        except:
            return False
        
    def has_also_view(self, idx, also_view_item):
        try: 
            also_buy_lst = self.get_neighbor_nodes(idx, 'also_view') 
            return also_view_item in also_buy_lst
        except:
            return False
    
# read review files
def read_review(path):
  def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
      yield json.loads(l)
  def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')
  return getDF(path)


# read qa files
def read_qa(path):
  def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
      yield eval(l)
  def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')
  return getDF(path)
