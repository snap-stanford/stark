def dict_tree(dictionary, indent=0):
    tree_str = ''
    for key, value in dictionary.items():
        if indent > 0:
            tree_str += '    |-----' * indent + f"{key}\n"
        else:
            tree_str += f"--{key}\n"
        if isinstance(value, dict):
            tree_str += dict_tree(value, indent + 1)
    return tree_str
    

class Node:
    def __init__(self):
        pass

    def __repr__(self):
        return dict_tree(self.dictionary)

    def __attr__(self):
        attributes = []
        lst = self.__dir__()
        for item in lst:
            if not item[:2] == '__':
                if isinstance(getattr(self, item), Node):
                    attributes.extend([f'{item}.{i}' for i in getattr(self, item).__attr__()])
                else:
                    attributes.append(item)
        return list(filter(lambda x: not 'dictionary' in x, attributes))
    
        
def register_node(node, dictionary):
    setattr(node, 'dictionary', dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            setattr(node, key, Node())
            register_node(getattr(node, key), value)
        else:
            setattr(node, key, value)


def df_row_to_dict(row, colunm_names=None):
    '''
    Convert a row of a dataframe to a dictionary.
    Args:
        row (pandas.Series): a row of a dataframe
    Return:
        dict: a dictionary that contains the same information as the row
    '''
    if colunm_names is None:
        colunm_names = row.columns
    return {name: row[name] for name in colunm_names}
