# c45_logic.py (Final Version: BIGGER FONTS)

import pandas as pd
import numpy as np
from math import log2
from graphviz import Digraph

# --- Helper function to format long labels ---
def format_label(label, max_length=25):
    """Adds newline characters to a string to wrap it for Graphviz."""
    words = label.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " "
            current_line += word
    if current_line:
        lines.append(current_line)
    return "\\n".join(lines)

# --- TreeNode Class ---
class TreeNode:
    def __init__(self, label=None, attribute=None, branches=None, N=0, E=0):
        self.label = label
        self.attribute = attribute
        self.branches = branches or {}
        self.N = N
        self.E = E

    def get_label(self):
        return str(self.label) if self.label is not None else str(self.attribute)

    def is_leaf(self):
        return not bool(self.branches)

    def add_to_dot(self, dot):
        node_id = str(id(self))
        
        raw_label = self.get_label()
        formatted_label = format_label(raw_label)

        # UPDATED: Increased fontsize to 14 for readability
        if self.is_leaf():
            dot.node(node_id, label=formatted_label, shape='box', style='rounded,filled', fillcolor='lightgrey', fontsize='14')
        else:
            dot.node(node_id, label=formatted_label, shape='ellipse', style='filled', fillcolor='lightblue', fontsize='14')
            
        for branch_value, branch in self.branches.items():
            branch.add_to_dot(dot)
            # UPDATED: Increased edge label fontsize to 12
            dot.edge(node_id, str(id(branch)), label=str(branch_value), fontsize='12')

# --- Core Helper Functions ---
def compute_entropy(labels):
    if len(labels) == 0: return 0
    probs = labels.value_counts() / len(labels)
    return -np.sum(probs * np.log2(probs))

def most_common_value(df, target_attribute):
    return df[target_attribute].mode()[0]

# --- Attribute Selection Logic ---
def choose_best_attribute(examples, target_attribute, attributes):
    best_gain_ratio = -1
    best_attribute = None
    best_threshold = None
    
    base_entropy = compute_entropy(examples[target_attribute])

    for attribute in attributes:
        subset_not_missing = examples.dropna(subset=[attribute])
        if len(subset_not_missing) == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(subset_not_missing[attribute])
        unique_vals = sorted(subset_not_missing[attribute].unique())
        
        # Integer-Binary Logic
        if is_numeric and 2 < len(unique_vals) <= 15:
            for i in range(len(unique_vals) - 1):
                threshold = unique_vals[i]
                
                left_subset = subset_not_missing[subset_not_missing[attribute] <= threshold]
                right_subset = subset_not_missing[subset_not_missing[attribute] > threshold]
                
                if len(left_subset) == 0 or len(right_subset) == 0:
                    continue

                p_left = len(left_subset) / len(subset_not_missing)
                p_right = len(right_subset) / len(subset_not_missing)
                
                info_gain = base_entropy - (p_left * compute_entropy(left_subset[target_attribute]) + p_right * compute_entropy(right_subset[target_attribute]))
                split_info = - (p_left * log2(p_left) + p_right * log2(p_right)) if p_left > 0 and p_right > 0 else 0
                
                gain_ratio = (info_gain / split_info if split_info != 0 else 0) * (len(subset_not_missing) / len(examples))
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_attribute = attribute
                    best_threshold = float(threshold)
        else: # Regular categorical
            values = unique_vals
            weighted_entropy = 0
            split_info = 0
            for value in values:
                subset = subset_not_missing[subset_not_missing[attribute] == value]
                weight = len(subset) / len(subset_not_missing)
                weighted_entropy += weight * compute_entropy(subset[target_attribute])
                if weight > 0: split_info -= weight * log2(weight)
            
            gain = base_entropy - weighted_entropy
            gain_ratio = (gain / split_info if split_info != 0 else 0) * (len(subset_not_missing) / len(examples))
            
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attribute
                best_threshold = None
                        
    return best_attribute, best_threshold

# --- C4.5 Tree Building ---
def c45(examples, target_attribute, attributes):
    root = TreeNode()
    
    if len(examples[target_attribute].unique()) == 1:
        label = examples[target_attribute].iloc[0]
        root.label = f'{label} ({len(examples):.1f}/0.0)'
        root.N, root.E = len(examples), 0
        return root

    if not attributes:
        label = most_common_value(examples, target_attribute)
        errors = len(examples[examples[target_attribute] != label])
        root.label = f'{label} ({len(examples):.1f}/{errors:.1f})'
        root.N, root.E = len(examples), errors
        return root

    best_attribute, threshold = choose_best_attribute(examples, target_attribute, attributes)

    if best_attribute is None:
        label = most_common_value(examples, target_attribute)
        errors = len(examples[examples[target_attribute] != label])
        root.label = f'{label} ({len(examples):.1f}/{errors:.1f})'
        root.N, root.E = len(examples), errors
        return root

    root.attribute = f"{best_attribute}"
    remaining_attrs = [attr for attr in attributes if attr != best_attribute]
    missing_data = examples[examples[best_attribute].isnull()]

    if threshold is not None: 
        subset_le = examples[examples[best_attribute] <= threshold]
        if not missing_data.empty: subset_le = pd.concat([subset_le, missing_data])
        if not subset_le.empty:
            root.branches[f"<= {threshold}"] = c45(subset_le, target_attribute, remaining_attrs)
        else:
            label = most_common_value(examples, target_attribute)
            root.branches[f"<= {threshold}"] = TreeNode(label=f'{label} (0.0/0.0)', N=0, E=0)

        subset_gt = examples[examples[best_attribute] > threshold]
        if not missing_data.empty: subset_gt = pd.concat([subset_gt, missing_data])
        if not subset_gt.empty:
            root.branches[f"> {threshold}"] = c45(subset_gt, target_attribute, remaining_attrs)
        else:
            label = most_common_value(examples, target_attribute)
            root.branches[f"> {threshold}"] = TreeNode(label=f'{label} (0.0/0.0)', N=0, E=0)

    else: 
        for value in sorted(examples[best_attribute].dropna().unique()):
            subset = examples[examples[best_attribute] == value]
            if not missing_data.empty: subset = pd.concat([subset, missing_data])
            if not subset.empty:
                root.branches[f"= {value}"] = c45(subset, target_attribute, remaining_attrs)
            else:
                label = most_common_value(examples, target_attribute)
                root.branches[f"= {value}"] = TreeNode(label=f'{label} (0.0/0.0)', N=0, E=0)
            
    return root

# --- Post-Pruning ---
def pessimistic_prune(node, target_attribute):
    if node.is_leaf(): return node
    for branch_value, subtree in node.branches.items():
        node.branches[branch_value] = pessimistic_prune(subtree, target_attribute)
    if not all(child.is_leaf() for child in node.branches.values()):
        return node
    
    subtree_N = sum(leaf.N for leaf in get_leaves(node))
    subtree_E = sum(leaf.E for leaf in get_leaves(node))
    pessimistic_subtree_error = subtree_E + 0.5 * len(node.branches)
    
    leaves_data = [(leaf.N, leaf.E, leaf.label.split('(')[0].strip()) for leaf in get_leaves(node)]
    class_counts = {}
    for N, E, label in leaves_data:
        class_counts[label] = class_counts.get(label, 0) + (N - E)
    
    if not class_counts: return node
    majority_class = max(class_counts, key=class_counts.get)
    pruned_E = subtree_N - class_counts[majority_class]
    pessimistic_pruned_error = pruned_E + 0.5
    
    if pessimistic_pruned_error <= pessimistic_subtree_error:
        new_label = f"{majority_class} ({subtree_N:.1f}/{pruned_E:.1f})"
        return TreeNode(label=new_label, N=subtree_N, E=pruned_E)
    else:
        return node

def get_leaves(node):
    if node.is_leaf(): return [node]
    leaves = []
    for branch in node.branches.values():
        leaves.extend(get_leaves(branch))
    return leaves

# --- Rule Generation ---
def generate_rules(node, rule_prefix="IF", rules=None):
    if rules is None: rules = []
    if node.is_leaf():
        full_label = node.label
        decision = full_label.split('(')[0].strip()
        stats = ""
        if '(' in full_label and ')' in full_label:
            stats_part = full_label.split('(')[1].replace(')', '')
            stats = f"[Stats: {stats_part}]"
        
        rules.append(f"{rule_prefix} THEN Class = {decision} {stats}")
        return rules

    for branch_value, subtree in node.branches.items():
        condition = f"{node.attribute} {branch_value}"
        new_prefix = f"{rule_prefix} {condition}" if rule_prefix == "IF" else f"{rule_prefix} AND {condition}"
        generate_rules(subtree, new_prefix, rules)
    return rules


