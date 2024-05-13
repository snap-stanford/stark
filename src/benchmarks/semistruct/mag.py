import os
import os.path as osp
import gzip
import pickle
import json
import torch
import pandas as pd
import numpy as np
from langdetect import detect
import gdown
from huggingface_hub import hf_hub_download
import zipfile
from ogb.nodeproppred import NodePropPredDataset
from ogb.utils.url import download_url, extract_zip
from src.tools.process_text import clean_data, compact_text, decode_escapes
from src.benchmarks.semistruct.knowledge_base import SemiStructureKB
from src.tools.io import save_files, load_files

PROCESSED_DATASET = {
    "repo": "snap-stanford/STaRK-Dataset",
    "file": "mag_processed.zip",
}

class MagSemiStruct(SemiStructureKB):
    
    test_columns = ['title', 'abstract', 'text']
    candidate_types = ['paper']

    node_type_dict =  {0: 'author', 1: 'institution', 2: 'field_of_study', 3: 'paper'}
    edge_type_dict = {
        0: 'author___affiliated_with___institution',
        1: 'paper___cites___paper', 
        2: 'paper___has_topic___field_of_study',
        3: 'author___writes___paper'
        }
    node_attr_dict = {'paper': ['title', 'abstract', 'publication date', 'venue'],
                      'author': ['name'],
                      'institution': ['name'],
                      'field_of_study': ['name']}
    ogbn_papers100M_cache_url = 'https://drive.google.com/uc?id=1BWHBIukoVsCsJ2kCRPKbXXrh_rHdluIp'
    ogbn_papers100M_url = 'https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip'
    mag_mapping_url = 'https://zenodo.org/records/2628216/files'
    
    def __init__(self, root, download_processed=True, **kwargs):
        '''
        Args:
            root (str): root directory to store the dataset folder
            download_processed (bool): whether to download the processed data
        '''

        self.root = root 
        schema_dir=osp.join(root, 'schema'),                       
        self.raw_data_dir = osp.join(self.root, 'raw')
        self.processed_data_dir = osp.join(self.root, 'processed')  
        self.graph_data_root = osp.join(self.raw_data_dir, 'ogbn_mag')
        self.text_root = osp.join(self.raw_data_dir, 'ogbn_papers100M')

        # existing dirs/files
        self.schema_dir = schema_dir
        self.mag_mapping_dir = osp.join(self.graph_data_root, 'mag_mapping')
        self.ogbn_mag_mapping_dir = osp.join(self.graph_data_root, 'mapping')
        self.title_path = osp.join(self.text_root, 'paperinfo/idx_title.tsv')
        self.abstract_path = osp.join(self.text_root, 'paperinfo/idx_abs.tsv')

        # new files
        self.mag_metadata_cache_dir = osp.join(self.processed_data_dir, 'mag_cache')
        self.paper100M_text_cache_dir = osp.join(self.processed_data_dir, 'paper100M_cache')
        self.merged_filtered_path = osp.join(self.paper100M_text_cache_dir, 'idx_title_abs.tsv')
        os.makedirs(self.mag_metadata_cache_dir, exist_ok=True)
        os.makedirs(self.paper100M_text_cache_dir, exist_ok=True)

        if not osp.exists(osp.join(self.processed_data_dir, 'node_info.pkl')) and download_processed:
            print('Downloading processed data...')
            processed_path = hf_hub_download(
                PROCESSED_DATASET["repo"],
                PROCESSED_DATASET["file"],
                repo_type="model",
            )
            with zipfile.ZipFile(processed_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(processed_path)
            print('Downloaded processed data!')

        if osp.exists(osp.join(self.processed_data_dir, 'node_info.pkl')):
            print(f'loaded processed data from {self.processed_data_dir}!')
            processed_data = load_files(self.processed_data_dir)
        else:
            print('start processing raw data')
            processed_data = self._process_raw()
        processed_data.update({'node_type_dict': self.node_type_dict, 
                               'edge_type_dict': self.edge_type_dict})
        super(MagSemiStruct, self).__init__(**processed_data, **kwargs)

    def load_edge(self, edge_type):
        edge_dir = osp.join(self.graph_data_root, f"raw/relations/{edge_type}/edge.csv.gz")
        edge_type_dir = osp.join(self.graph_data_root, f"raw/relations/{edge_type}/edge_reltype.csv.gz")
        num_dir = osp.join(self.graph_data_root, f"raw/relations/{edge_type}/num-edge-list.csv.gz")
        edge = pd.read_csv(edge_dir, names=['src', 'dst'])
        
        edge_t = pd.read_csv(edge_type_dir, names=['type'])
        edge_n = pd.read_csv(num_dir, names=['num'])
        edge_num = edge_n['num'].tolist()

        edge = [edge['src'].tolist(), edge['dst'].tolist(), edge_t['type'].tolist()]
        edge = torch.LongTensor(edge)

        return edge, edge_num
    
    def load_meta_data(self):
        
        mag_csv = {}
        if osp.exists(osp.join(self.mag_metadata_cache_dir, 'paper_data.csv')):
            print('start loading MAG data from cache')
            for t in ['author', 'institution', 'field_of_study', 'paper']:
                mag_csv[t] = pd.read_csv(osp.join(self.mag_metadata_cache_dir, f'{t}_data.csv'))
            author_data, paper_data = mag_csv['author'], mag_csv['paper']
            field_of_study_data = mag_csv['field_of_study']
            institution_data = mag_csv['institution']
            print('done!')
        else:
            print('start loading MAG data, it might take a while...')
            full_attr_path = osp.join(self.schema_dir, 'mag.json')
            reduced_attr_path = osp.join(self.schema_dir, 'reduced_mag.json')

            full_attr = json.load(open(full_attr_path, 'r'))
            reduced_attr = json.load(open(reduced_attr_path, 'r'))

            loaded_csv = {}
            for key in reduced_attr.keys():
                column_nums = [full_attr[key].index(i) for i in reduced_attr[key]]
                file = osp.join(self.mag_mapping_dir, key + '.txt.gz')
                if not osp.exists(file):
                    try:
                        download_url(f'{self.mag_mapping_url}/{key}.txt.gz', self.mag_mapping_dir)
                    except Exception as error:
                        print(f'Download failed or {key} data not found, please download from {self.mag_mapping_url} to {file}')
                        raise error
                loaded_csv[key] = pd.read_csv(file, header=None, sep='\t', usecols=column_nums)
                loaded_csv[key].columns = reduced_attr[key]

            print('processing and merging meta data...')
            author_data = pd.read_csv(osp.join(self.ogbn_mag_mapping_dir, f"author_entidx2name.csv.gz"), names=['id', 'AuthorId'], skiprows=[0])
            field_of_study_data = pd.read_csv(osp.join(self.ogbn_mag_mapping_dir, f"field_of_study_entidx2name.csv.gz"), names=['id', 'FieldOfStudyId'], skiprows=[0])
            institution_data = pd.read_csv(osp.join(self.ogbn_mag_mapping_dir, f"institution_entidx2name.csv.gz"), names=['id', 'AffiliationId'], skiprows=[0])
            paper_data = pd.read_csv(osp.join(self.ogbn_mag_mapping_dir, f"paper_entidx2name.csv.gz"), names=['id', 'PaperId'], skiprows=[0])

            loaded_csv['Papers'].rename(columns={'JournalId ': 'JournalId', 'Rank': 'PaperRank', 'CitationCount': 'PaperCitationCount'}, inplace=True)
            loaded_csv['Journals'].rename(columns={'DisplayName': 'JournalDisplayName', 'Rank': 'JournalRank', 'CitationCount': 'JournalCitationCount', 'PaperCount': 'JournalPaperCount'}, inplace=True)
            loaded_csv['ConferenceSeries'].rename(columns={'DisplayName': 'ConferenceSeriesDisplayName', 'Rank': 'ConferenceSeriesRank', 'CitationCount': 'ConferenceSeriesCitationCount', 'PaperCount': 'ConferenceSeriesPaperCount'}, inplace=True)
            loaded_csv['ConferenceInstances'].rename(columns={'DisplayName': 'ConferenceInstancesDisplayName', 'CitationCount': 'ConferenceInstanceCitationCount', 'PaperCount': 'ConferenceInstancesPaperCount'}, inplace=True)

            author_data = author_data.merge(loaded_csv['Authors'], on='AuthorId', how='left')
            field_of_study_data = field_of_study_data.merge(loaded_csv['FieldsOfStudy'], on='FieldOfStudyId', how='left')
            institution_data = institution_data.merge(loaded_csv['Affiliations'], on='AffiliationId', how='left')
            paper_data = paper_data.merge(loaded_csv['Papers'], on='PaperId', how='left')

            paper_data['JournalId'] = paper_data['JournalId'].apply(lambda x: float(x)).apply(lambda x: -1 if np.isnan(x) else int(x))
            paper_data = paper_data.merge(loaded_csv['Journals'], on='JournalId', how='left')

            paper_data = paper_data.merge(loaded_csv['ConferenceSeries'], on='ConferenceSeriesId', how='left')

            paper_data['ConferenceInstanceId'] = paper_data['ConferenceInstanceId'].apply(lambda x: float(x)).apply(lambda x: -1 if np.isnan(x) else int(x))
            paper_data = paper_data.merge(loaded_csv['ConferenceInstances'], on='ConferenceInstanceId', how='left')

            for csv_data in [author_data, field_of_study_data, institution_data, paper_data]:
                csv_data.columns = csv_data.columns.str.strip()
                for col in csv_data.columns:
                    csv_data[col] = csv_data[col].apply(lambda x: -1 if isinstance(x, float) and np.isnan(x) else x)
                    if 'rank' in col.lower() or 'count' in col.lower() or 'level' in col.lower() or 'year' in col.lower() or col.lower().endswith('id'):
                        csv_data[col] = csv_data[col].apply(lambda x: int(x) if isinstance(x, float) else x)
            
            mag_csv = {'author': author_data, 
                       'institution': institution_data, 
                       'field_of_study': field_of_study_data, 
                       'paper': paper_data}
            
            for t in ['author', 'institution', 'field_of_study', 'paper']:
                mag_csv[t].to_csv(osp.join(self.mag_metadata_cache_dir, f'{t}_data.csv'), index=False)
            author_data, paper_data = mag_csv['author'], mag_csv['paper']
            field_of_study_data = mag_csv['field_of_study']
            institution_data = mag_csv['institution']

        # create init_id to mag_id mapping
        author_data['type'] = 'author'
        author_data.rename(columns={'id': 'id', 'AuthorId': 'mag_id'}, inplace=True)

        institution_data['type'] = 'institution'
        institution_data.rename(columns={'id': 'id', 'AffiliationId': 'mag_id'}, inplace=True)

        field_of_study_data['type'] = 'field_of_study'
        field_of_study_data.rename(columns={'id': 'id', 'FieldOfStudyId': 'mag_id'}, inplace=True)

        paper_data['type'] = 'paper'
        paper_data.rename(columns={'id': 'id', 'PaperId': 'mag_id'}, inplace=True)
        return author_data, field_of_study_data, institution_data, paper_data

    def load_english_paper_text(self, mag_ids, download_cache=True):
        def is_english(text):
            try:
                return detect(text) == 'en'
            except:
                return False
        if not osp.exists(self.merged_filtered_path):
            if download_cache:
                # We provided cache here to avoid processing the large file for a long time
                try:
                    gdown.download(self.ogbn_papers100M_cache_url, 
                                   self.merged_filtered_path, quiet=False)
                except Exception as error:
                    print('Try upgrading your gdown package with `pip install gdown --upgrade`')
                    raise error
            else:
                if not osp.exists(self.title_path):  
                    raw_text_path = download_url(self.ogbn_papers100M_url, self.text_root)
                    extract_zip(raw_text_path, self.text_root)
                print('start read title...')
                title = pd.read_csv(self.title_path, sep='\t', header=None)
                title.columns = ["mag_id", "title"]
                print('filtering title in English...')

                # filter the title that's in mag_ids
                title = title[title['mag_id'].apply(lambda x: x in mag_ids)]
                title_en = title[title['title'].apply(is_english)]

                print('start read abstract...')
                abstract = pd.read_csv(self.abstract_path, sep='\t', header=None)
                abstract.columns = ["mag_id", "abstract"]
                print('filtering abstract in English...')

                abstract = abstract[abstract['mag_id'].apply(lambda x: x in mag_ids)]
                abstract_en = abstract[abstract['abstract'].apply(is_english)]

                print('start merging title and abstract...')
                title_abs_en = pd.merge(title, abs, how="outer", on="mag_id", sort=True)
                title_abs_en.to_csv(self.merged_filtered_path, sep="\t", header=True, index=False)
                
        print('loading merged and filtered title and abstract (English)...')
        title_abs_en = pd.read_csv(self.merged_filtered_path, sep='\t')
        title_abs_en.columns = ['mag_id', 'title', 'abstract']
        print('done!')

        return title_abs_en

    def get_map(self, df):
        mag2id, id2mag = {}, {}
        for idx in range(len(df)):
            mag2id[df['mag_id'][idx]] = idx
            id2mag[idx] = df['mag_id'][idx]
        return mag2id, id2mag
    
    def get_doc_info(self, idx, compact=False, 
                     add_rel=True, n_rel=-1) -> str:
        node = self[idx]
        if node.type == 'author':
            doc = f'- author name: {node.DisplayName}\n'
            if node.PaperCount != -1:
                doc += f'- author paper count: {node.PaperCount}\n'
            if node.CitationCount != -1:
                doc += f'- author citation count: {node.CitationCount}\n'
            doc = doc.replace('-1', 'Unknown')

        elif node.type == 'paper':
            doc = ' - paper title: ' + node.title + '\n'
            doc += ' - abstract: ' + node.abstract.replace('\r', '').rstrip('\n') + '\n'
            if str(node.Date) != '-1':
                doc += ' - publication date: ' + str(node.Date) + '\n'
            if str(node.OriginalVenue) != '-1':
                doc += ' - venue: ' + node.OriginalVenue + '\n'
            elif str(node.JournalDisplayName) != '-1':
                doc += ' - journal: ' + node.JournalDisplayName + '\n'
            elif str(node.ConferenceSeriesDisplayName) != '-1':
                doc += ' - conference: ' + node.ConferenceSeriesDisplayName + '\n'
            elif str(node.ConferenceInstancesDisplayName) != '-1':
                doc += ' - conference: ' + node.ConferenceInstancesDisplayName + '\n'

        elif node.type == 'field_of_study':
            doc = ' - field of study: ' + node.DisplayName + '\n' 
            if node.PaperCount != -1:
                doc += f'- field paper count: {node.PaperCount}\n'
            if node.CitationCount != -1:
                doc += f'- field citation count: {node.CitationCount}\n'
            doc = doc.replace('-1', 'Unknown')
            
        elif node.type == 'institution':
            doc = ' - institution: ' + node.DisplayName + '\n' 
            if node.PaperCount != -1:
                doc += f'- institution paper count: {node.PaperCount}\n'
            if node.CitationCount != -1:
                doc += f'- institution citation count: {node.CitationCount}\n'
            doc = doc.replace('-1', 'Unknown')
        
        if add_rel and node.type == 'paper':
            doc += self.get_rel_info(idx, n_rel=n_rel)

        if compact:
            doc = compact_text(doc)
        return doc

    def get_rel_info(self, idx, rel_types=None, n_rel=-1):
        doc = ''
        rel_types = self.rel_type_lst() if rel_types is None else rel_types
        for edge_t in rel_types:
            node_ids = torch.LongTensor(self.get_neighbor_nodes(idx, edge_t)).tolist()
            if len(node_ids) == 0: 
                continue
            node_type = self.node_types[node_ids[0]]
            str_edge = edge_t.replace('___', ' ')
            doc += f"\n{str_edge}: "
            
            if n_rel > 0 and edge_t == 'paper___cites___paper':
                node_ids = node_ids[torch.randperm(len(node_ids))[:n_rel]].tolist()
            neighbors = []
            for i in node_ids:
                if self[i].type == 'paper':
                    neighbors.append(f'\"{self[i].title}\"')
                elif self[i].type == 'author':
                    if not str(self[i].DisplayName) == '-1':
                        institutions = self.get_neighbor_nodes(i, "author___affiliated_with___institution")
                        for inst in institutions:
                            assert self[inst].type == 'institution'
                        str_institutions = [self[j].DisplayName for j in institutions if not str(self[j].DisplayName) == '-1']
                        if len(str_institutions) > 0:
                            str_institutions = ', '.join(str_institutions)
                            neighbors.append(f'{self[i].DisplayName} ({str_institutions})')
                        else:
                            neighbors.append(f'{self[i].DisplayName}')
                else:
                    if not str(self[i].DisplayName) == '-1':
                        neighbors.append(f'{self[i].DisplayName}')
            neighbors = '(' + ', '.join(neighbors) + '),'
            doc += neighbors
        if len(doc): 
            doc = '- relations:\n' + doc
        return doc 
    
    def _process_raw(self):
        NodePropPredDataset(name='ogbn-mag', root=self.raw_data_dir)
        author_data, field_of_study_data, institution_data, paper_data = self.load_meta_data()
        paper_text_data = self.load_english_paper_text(paper_data['mag_id'].tolist())

        print('precessing graph data...')
        author_id_to_mag = {row['id']: row['mag_id'] for _, row in author_data.iterrows()}
        institution_id_to_mag = {row['id']: row['mag_id'] for _, row in institution_data.iterrows()}
        field_of_study_id_to_mag = {row['id']: row['mag_id'] for _, row in field_of_study_data.iterrows()}
        paper_mapping = pd.read_csv(osp.join(self.ogbn_mag_mapping_dir, f"paper_entidx2name.csv.gz"), names=['id', 'mag_id'], skiprows=[0])
        mag_to_paper_id, paper_id_to_mag = self.get_map(paper_mapping)

        unique_paper_id = paper_text_data['mag_id'].unique()
        unique_paper_id = torch.unique(torch.tensor(unique_paper_id))
        node_type_edge = {
            0:'author___writes___paper', 
            2:'paper___has_topic___field_of_study', 
            3:'paper___cites___paper'}
        node_type_overlapping_node = {}
        node_type_overlapping_edge = {}

        # from mag_id to id
        for k, v in mag_to_paper_id.items():
            if k not in unique_paper_id:
                continue
            mask = unique_paper_id == k
            unique_paper_id[mask] = v

        # load edge data
        print('start loading edge data')
        for node_type, paper_rel in node_type_edge.items():
            print(node_type, paper_rel)
            edge, edge_num = self.load_edge(paper_rel)
            # Identify edges connected to target nodes
            if node_type == 3:
                target_array = unique_paper_id.numpy()
                edge_array = edge.numpy()
                mask = np.isin(edge_array[0], target_array) & np.isin(edge_array[1], target_array)
                valid_edges_array = edge_array[:, mask]
                valid_edges_tensor = torch.from_numpy(valid_edges_array)
                node_type_overlapping_node[node_type] = unique_paper_id
                node_type_overlapping_edge[node_type] = valid_edges_tensor
                print(f'{node_type} has {unique_paper_id.shape[0]} nodes left,  and {valid_edges_tensor.t().shape[0]} edges left.')
                continue
            else:
                edge = edge.t()
                connected_edges_list = [] 
                for target_node in unique_paper_id:
                    # Find the edges connected to the current target node
                    if node_type == 0:
                        mask = edge[:, 1] == target_node.item()
                        current_connected_edges = edge[mask].clone() 
                    elif node_type == 2:
                        mask = edge[:, 0] == target_node.item()
                        current_connected_edges = edge[mask].clone() 
                    
                    # Collect the other ends of the connected edges
                    connected_edges_list.append(current_connected_edges)
                    del mask
                    del current_connected_edges
                    # print(len(connected_edges_list))

                connected_edges = torch.cat(connected_edges_list, dim=0)
                if node_type == 0:
                    other_ends = torch.unique(connected_edges.t()[0])
                elif node_type == 2:
                    other_ends = torch.unique(connected_edges.t()[1])

                node_type_overlapping_node[node_type] = other_ends
                node_type_overlapping_edge[node_type] = connected_edges.t()
                print(f'{node_type} has {other_ends.shape[0]} nodes left,  and {connected_edges.shape[0]} edges left.')

        # specifically choose for institution by author
        edge, edge_num = self.load_edge('author___affiliated_with___institution')
        edge = edge.t()
        connected_edges_list = []
        for target_node in node_type_overlapping_node[0]:
            mask = edge[:, 0] == target_node
            current_connected_edges = edge[mask].clone()
            # Collect the other ends of the connected edges
            connected_edges_list.append(current_connected_edges)

        connected_edges = torch.cat(connected_edges_list, dim=0)
        other_ends = torch.unique(connected_edges.t()[1])

        node_type_overlapping_node[1] = other_ends
        node_type_overlapping_edge[1] = connected_edges.t()
        print(f'1 has {other_ends.shape[0]} nodes left,  and {connected_edges.shape[0]} edges left.')

        # save shared nodes in node_type_overlapping_node and shared edges in node_type_overlapping_edge
        tot_n = sum([len(node_type_overlapping_node[i]) for i in range(4)])

        # the order of re-indexing is author, institution, field_of_study, paper
        domain_mappings = {0: author_id_to_mag, 
                           1: institution_id_to_mag, 
                           2: field_of_study_id_to_mag, 
                           3: paper_id_to_mag}
        new_domain_mappings = {}
        domain_old_to_new = {}
        id_to_mag = {}
        offset = 0
        node_type_overlapping_node_sort = {k: node_type_overlapping_node[k] for k in sorted(node_type_overlapping_node.keys())}

        # start to re-index 
        print('start re-indexing')
        for i, remain_node in node_type_overlapping_node_sort.items():
            old_to_new_mappings = {key: id + offset for id, key in enumerate(remain_node.tolist())}
            updated_dict = {value: domain_mappings[i][key] for key, value in old_to_new_mappings.items()}
            print(f'{i} has {len(updated_dict)} nodes left')
            domain_old_to_new[i] = old_to_new_mappings
            id_to_mag.update(updated_dict)
            new_domain_mappings[i] = updated_dict
            offset += len(node_type_overlapping_node[i])

        # check last index equals tot_n
        assert offset == tot_n
        edges_full = torch.cat([node_type_overlapping_edge[i] for i in range(4)], dim=1)

        # re-index edges 
        # Different types of nodes all start from 0, need to re-index according to types
        d_of_mapping_dict = {
            0: [domain_old_to_new[0], domain_old_to_new[3]], 
            1: [domain_old_to_new[0], domain_old_to_new[1]], 
            2: [domain_old_to_new[3], domain_old_to_new[2]], 
            3: [domain_old_to_new[3], domain_old_to_new[3]]}

        for i, remain_edge in node_type_overlapping_edge.items():
            edges = remain_edge[:2]
            edge_types = remain_edge[2]
            new_edges = edges.clone()
            dict1 = d_of_mapping_dict[i][0]
            dict2 = d_of_mapping_dict[i][1]

            # Update the first dimension using dict1
            for old, new in dict1.items():
                new_edges[0, edges[0] == old] = new

            # Update the second dimension using dict2
            for old, new in dict2.items():
                new_edges[1, edges[1] == old] = new

            final_edges = torch.cat([new_edges, edge_types.unsqueeze(0)], dim=0)
            node_type_overlapping_edge[i] = final_edges

        edges_final = torch.cat([node_type_overlapping_edge[i] for i in range(4)], dim=1)
        assert edges_final.shape == edges_full.shape
        edge_index = torch.LongTensor(edges_final[:2])
        edge_types = torch.LongTensor(edges_final[2])

        # re-index nodes
        author_data['new_id'] = author_data['id'].map(domain_old_to_new[0])
        author_data.dropna(subset=['new_id'], inplace=True)
        author_data['new_id'] = author_data['new_id'].astype(int)
        institution_data['new_id'] = institution_data['id'].map(domain_old_to_new[1])
        institution_data.dropna(subset=['new_id'], inplace=True)
        institution_data['new_id'] = institution_data['new_id'].astype(int)
        field_of_study_data['new_id'] = field_of_study_data['id'].map(domain_old_to_new[2])
        field_of_study_data.dropna(subset=['new_id'], inplace=True)
        field_of_study_data['new_id'] = field_of_study_data['new_id'].astype(int)
        paper_data['new_id'] = paper_data['id'].map(domain_old_to_new[3])
        paper_data.dropna(subset=['new_id'], inplace=True)
        paper_data['new_id'] = paper_data['new_id'].astype(int)

        # add text data onto the graph(paper nodes)
        merged_df = pd.merge(paper_data, paper_text_data, on='mag_id', how='outer')
        merged_df.dropna(subset=['new_id'], inplace=True)
        merged_df['new_id'] = merged_df['new_id'].astype(int)
        merged_df['mag_id'] = merged_df['mag_id'].astype(int)
        merged_df = merged_df.drop_duplicates(subset=['new_id'])

        # record node_info into dict
        node_frame = {0: author_data, 1: institution_data, 2: field_of_study_data, 3: merged_df}
        node_info = {}
        node_types = []
        for node_type, frame in node_frame.items():
            for idx, row in frame.iterrows():
                # csv_row to dict
                node_info[row['new_id']] = row.to_dict()
                node_types.append(node_type)
        node_types = torch.tensor(node_types)
        if len(node_types) != tot_n:
            raise ValueError('node_types length does not match tot_n')

        processed_data = {
            'node_info': node_info, 
            'edge_index': edge_index, 
            'edge_types': edge_types,
            'node_types': node_types
            }

        print('start saving processed data')
        save_files(save_path=self.processed_data_dir, **processed_data)

        return processed_data