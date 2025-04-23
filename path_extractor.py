import pandas as pd
from anytree import Node, RenderTree
from anytree.search import findall_by_attr
from anytree.walker import Walker
import numpy as np
import random
import javalang
from config import Config

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node 
    elif isinstance(node, set):
        token = 'Modifier'
        # node.pop()
    elif isinstance(node, javalang.ast.Node):
        token = node.__class__.__name__  
        

    return token

def get_children(root):
    if isinstance(root, javalang.ast.Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

        
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
                
    return list(expand(children))

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order, token], parent=parent_node, order=order)

    for child_order in range(len(children)):
        get_trees(children[child_order], node, order+str(int(child_order)+1))

def get_path_length(path):
    """Calculating path length.
    Input:
    path: list. Containing full walk path.

    Return:
    int. Length of the path.
    """
    
    return len(path)

def get_path_width(raw_path):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    int. Width of the path.
    """
    
    return abs(int(raw_path[0][-1].order)-int(raw_path[2][0].order))
    
def hashing_path(path, hash_table):
    """Calculating path width.
    Input:
    # raw_path: tuple. Containing upstream, parent, downstream of the path.
    text_path：str. 用@连接的完整路径字符串

    Return:
    str. Hash of the path.
    """
    
    if path not in hash_table:
        hash = random.getrandbits(128)
        hash_table[path] = str(hash)
        return str(hash)
    else:
        return hash_table[path]

def get_node_rank(node_name, max_depth):
    """Calculating node rank for leaf nodes.
    Input:
    node_name: list. where the first element is the string order of the node, second element is actual name.
    max_depth: int. the max depth of the code.

    Return:
    list. updated node name list.
    """
    while len(node_name[0]) < max_depth:
        node_name[0] += "0" 
    return [int(node_name[0]),node_name[1]]

def extracting_path(java_code, max_length, max_width, hash_path, hashing_table):
    """Extracting paths for a given json code.
    Input:
    java_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.

    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    head = Node(["1",get_token(java_code)])
    # head:  Node("/['1', 'MethodDeclaration']")
    
    for child_order in range(len(get_children(java_code))):
        get_trees(get_children(java_code)[child_order], head, "1"+str(int(child_order)+1))
    
    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
    
    max_depth = max([len(node.name[0]) for node in leaf_nodes])
    
    for leaf in leaf_nodes:
        leaf.name = get_node_rank(leaf.name, max_depth)
    
    walker = Walker()
    text_paths = []
    
    for leaf_index in range(len(leaf_nodes)-1):
        for target_index in range(leaf_index+1, len(leaf_nodes)):
            raw_path = walker.walk(leaf_nodes[leaf_index], leaf_nodes[target_index])
            
            walk_path = [n.name[1] for n in list(raw_path[0])]+[raw_path[1].name[1]]+[n.name[1] for n in list(raw_path[2])]
            text_path = "@".join(walk_path)
            
            if get_path_length(walk_path) <= max_length and get_path_width(raw_path) <= max_width:
                if not hash_path:
                    text_paths.append(walk_path[0]+",~,"+text_path+",~,"+walk_path[-1])
                else:
                    text_paths.append(walk_path[0]+",~,"+hashing_path(text_path, hashing_table)+",~,"+walk_path[-1])
    
    
    return text_paths


def program_parser(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    
    return tree


def main():
    
    config = Config()

    code_df = pd.read_csv("../Dataset/CodeWorkout/CodeStates.csv")
    main_df = pd.read_csv('../Dataset/CodeWorkout/MainTable.csv')
    
    main_df = main_df[main_df["EventType"] == "Run.Program"]
    main_df = main_df[main_df["AssignmentID"] == config.assignment]

    main_df['Score'] = np.array(main_df["Score"] == 1).astype(int)
    
    main_df = main_df.merge(code_df, left_on="CodeStateID", right_on="CodeStateID") 
    
    parsed_code = []
    for c in list(main_df['Code']):
        try:
            parsed = program_parser(c) 
        except:
            parsed = "Uncompilable"
        parsed_code.append(parsed)
    
    """
    MethodDeclaration(annotations=[], body=[IfStatement(condition=BinaryOperation(operandl=MemberReference(member=isWeekend, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=Literal(postfix_operators=[], prefix_operators=[], qualifier=None, selectors=[], value=true), operator===), else_statement=BlockStatement(label=None, statements=[ReturnStatement(expression=BinaryOperation(operandl=BinaryOperation(operandl=MemberReference(member=cigers, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=Literal(postfix_operators=[], prefix_operators=[], qualifier=None, selectors=[], value=40), operator=>=), operandr=BinaryOperation(operandl=MemberReference(member=cigers, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=Literal(postfix_operators=[], prefix_operators=[], qualifier=None, selectors=[], value=60), operator=<=), operator=&&), label=None)]), label=None, then_statement=BlockStatement(label=None, statements=[ReturnStatement(expression=BinaryOperation(operandl=MemberReference(member=cigers, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=Literal(postfix_operators=[], prefix_operators=[], qualifier=None, selectors=[], value=40), operator=>=), label=None)]))], documentation=None, modifiers={'public'}, name=cigarParty, parameters=[FormalParameter(annotations=[], modifiers=set(), name=cigars, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=isWeekend, type=BasicType(dimensions=[], name=boolean), varargs=False)], return_type=BasicType(dimensions=[], name=boolean), throws=None, type_parameters=None)
    """

    hashing_table = {}
    #hashing_table = np.load("path_hashing_dict.npy",allow_pickle=True).item()

    AST_paths = [extracting_path(java_code, max_length=config.code_path_length, max_width=config.code_path_width, hash_path=True, hashing_table=hashing_table) for java_code in parsed_code]
    
    main_df["RawASTPath"] = ["@~@".join(A) for A in AST_paths]
    
    main_df.to_csv("./data/labeled_paths.tsv", sep="\t", header=True)
    
    # return main_df
        
if __name__ == "__main__":
    
    main()
    
