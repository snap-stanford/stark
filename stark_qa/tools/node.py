def dict_tree(dictionary, indent=0):
    """
    Create a visual tree representation of a dictionary.

    Args:
        dictionary (dict): The dictionary to represent as a tree.
        indent (int): The current indentation level.

    Returns:
        str: A string representing the dictionary as a tree.
    """
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
        """
        Recursively gather attributes of the node.

        Returns:
            list: A list of attribute names.
        """
        attributes = []
        lst = self.__dir__()
        for item in lst:
            if not item.startswith('__'):
                if isinstance(getattr(self, item), Node):
                    attributes.extend([f'{item}.{i}' for i in getattr(self, item).__attr__()])
                else:
                    attributes.append(item)
        return list(filter(lambda x: 'dictionary' not in x, attributes))


def register_node(node, dictionary):
    """
    Register a dictionary into a Node object.

    Args:
        node (Node): The node to register the dictionary to.
        dictionary (dict): The dictionary to register.
    """
    setattr(node, 'dictionary', dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            setattr(node, key, Node())
            register_node(getattr(node, key), value)
        else:
            setattr(node, key, value)


def df_row_to_dict(row, column_names=None):
    """
    Convert a row of a DataFrame to a dictionary.

    Args:
        row (pandas.Series): A row of a DataFrame.
        column_names (list, optional): The list of column names. Defaults to None.

    Returns:
        dict: A dictionary that contains the same information as the row.
    """
    if column_names is None:
        column_names = row.index
    return {name: row[name] for name in column_names}
