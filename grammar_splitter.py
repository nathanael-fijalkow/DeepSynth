from pcfg import PCFG
import bisect
import numpy as np
from typing import Dict, List, Optional, Tuple
import bisect
from dataclasses import dataclass, field
import copy

from program import Program
from type_system import PrimitiveType, Type



@dataclass(frozen=True)
class Context:
    type: Type
    predecessors: List[Tuple[PrimitiveType, int]] = field(default_factory=lambda: [])
    depth: int = field(default=0)

@dataclass(order=True, frozen=True)
class Node:
    probability: float
    next_contexts: List[Context] = field(compare=False)
    program: List[Program] = field(compare=False)
    derivation_history: List[Context] = field(compare=False)


PRules = Dict[Context, Dict[Program, Tuple[List[Context], float]]]

def __common_prefix__(a: List[Context], b: List[Context]) -> List[Context]:
    if a == b:
        return a
    candidates = []
    if len(a) > 1:
        candidates.append(__common_prefix__(a[1:], b))
        if len(b) >= 1 and a[0] == b[0]:
            candidates.append([a[0]] + __common_prefix__(a[1:], b[1:]))
    if len(b) > 1:
        candidates.append(__common_prefix__(a, b[1:]))
    # Take longest common prefix
    lentghs = [len(x) for x in candidates]
    if len(lentghs) == 0:
        return []
    if max(lentghs) == lentghs[0]:
        return candidates[0]
    return candidates[1]


def __adapt_ctx__(S: Context, i: int) -> Context:
    pred = S.predecessors[0]
    return Context(S.type, [(pred[0], i)], S.depth)

def __to_original__(S: Context) -> Tuple:
    return (S.type, S.predecessors[0] if S.predecessors else None, S.depth)

def __from_original__(S: Tuple) -> Context:
    return  Context(S[0], [S[1]] if S[1] else [], S[2])

def __create_path__(
    rules: PRules,
    original_pcfg: PCFG,
    rule_no: int,
    Slist: List[Context],
    Plist: List[Program],
    mapping: Dict[Tuple, Context],
    original_start: Context,
) -> int:
    for i, (S, P) in enumerate(zip(Slist, Plist)):
        if i == 0:
            S = original_start
        derivations = original_pcfg.rules[__to_original__(S)][P][0]
        # Update derivations
        new_derivations = []
        for nS in derivations:
            cnS = __from_original__(nS)
            if cnS not in Slist:
                new_derivations.append(nS)
            else:
                if nS in mapping:
                    new_derivations.append(__to_original__(mapping[nS]))
                else:
                    mS = __adapt_ctx__(cnS, rule_no)
                    mapping[nS] = mS
                    new_derivations.append(__to_original__(mS))
                    rule_no += 1
        derivations = new_derivations
        # Update current S
        if i > 0:
            S = mapping[__to_original__(S)]
        else:
            S = Slist[0]
        # Add rule
        rules[__to_original__(S)] = {}
        rules[__to_original__(S)][P] = derivations, 1
    return rule_no


def __pcfg_from__(original_pcfg: PCFG, group: List[Node]) -> PCFG:
    # Find the common prefix to all
    min_prefix = copy.deepcopy(group[0].derivation_history)
    for node in group[1:]:
        min_prefix = __common_prefix__(min_prefix, node.derivation_history)

    # Extract the start symbol
    start = min_prefix.pop()

    rules: Dict[Context, Dict[Program, Tuple[List[Context], float]]] = {}
    rule_no: int = (
        max(
            key[1][1] if key[1] else 0
            for key in original_pcfg.rules.keys()
        )
        + 1
    )
    mapping: Dict[Tuple, Context] = {}
    # Our min_prefix may be something like (int, 1, (+, 1))
    # which means we already chose +
    # But it is not in the PCFG
    # Thus we need to add it
    # In the general case we may as well have + -> + -> + as prefix this whole prefix needs to be added
    original_start = start
    if len(min_prefix) > 0:
        Slist = group[0].derivation_history[: len(min_prefix) + 1]
        Plist = group[0].program[: len(min_prefix) + 1]
        print("HERE")
        rule_no = __create_path__(
            rules, original_pcfg, rule_no, Slist, Plist, mapping, Slist[0]
        )
        original_start = Slist[-1]
        start = mapping[__to_original__(original_start)]

    # Now we need to make a path from the common prefix to each node's prefix
    # We also need to mark all contexts that should be filled
    to_fill: List[Context] = []
    for node in group:
        args, program, prefix = (
            node.next_contexts,
            node.program,
            node.derivation_history,
        )
        # Create rules to follow the path
        i = prefix.index(original_start)
        ctx_path = prefix[i:]
        program_path = program[i:]
        if len(ctx_path) > 0:
            ctx_path[0] = start
            print("THERE", flush=True)
            rule_no = __create_path__(
                rules,
                original_pcfg,
                rule_no,
                ctx_path,
                program_path,
                mapping,
                original_start,
            )
        # If there is no further derivation
        if not args:
            continue
        # Next derivation should be filled
        for arg in args:
            to_fill.append(arg)

    # At this point rules can generate all partial programs
    # Get the S to normalize by descending depth order
    to_normalise = sorted(list(rules.keys()), key=lambda x: -x[2])

    # Build rules from to_fill
    while to_fill:
        S = to_fill.pop()
        rules[S] = {}
        for P in original_pcfg.rules[S]:
            args, w = original_pcfg.rules[S][P]
            rules[S][P] = (args[:], w)  # copy list
            for arg in args:
                if arg not in rules:
                    to_fill.append(arg)
    # At this point we have all the needed rules
    # However, the probabilites are incorrect
    while to_normalise:
        S = to_normalise.pop()
        if S not in original_pcfg.rules:
            continue
        # Compute the updated probabilities
        for P in list(rules[S]):
            args, _ = rules[S][P]
            # We have the following equation:
            # (1) w = old_w * remaining_fraction
            old_w = original_pcfg.rules[S][P][1]
            remaining_fraction: float = 1
            # If there is a next derivation use it to compute the remaining_fraction
            if args:
                N = args[-1]
                remaining_fraction = sum(rules[N][K][1] for K in rules[N])
            # Update according to Equation (1)
            rules[S][P] = args, old_w * remaining_fraction

        # The updated probabilities may not sum to 1 so we need to normalise them
        # But let PCFG do it with clean=True

    start = original_pcfg.start

    # Ensure rules are depth ordered
    rules = {
        key: rules[key] for key in sorted(list(rules.keys()), key=lambda x: x[2])
    }

    return PCFG(start, rules, original_pcfg.max_program_depth, clean=True)


def __node_split__(pcfg: PCFG, node: Node) -> Tuple[bool, List[Node]]:
    """
    Split the specified node accordingly.

    Return: success, nodes:
    - True, list of children nodes
    - False, [node]
    """
    output: List[Node] = []
    next_contexts = node.next_contexts
    # If there is no next then it means this node can't be split
    if len(next_contexts) == 0:
        return False, [node]
    new_context: Context = next_contexts.pop()
    for P in pcfg.rules[new_context]:
        args, p_prob = pcfg.rules[new_context][P]
        new_root = Node(
            node.probability * p_prob,
            next_contexts + args,
            node.program + [P],
            node.derivation_history + [__from_original__(new_context)],
        )
        output.append(new_root)
    return True, output


def __split_nodes_until_quantity_reached__(
    pcfg: PCFG, quantity: int
) -> List[Node]:
    """
    Start from the root node and split most probable node until the threshold number of nodes is reached.
    """
    nodes: List[Node] = [Node(1.0, [pcfg.start], [], [])]
    while len(nodes) < quantity:
        i = 1
        success, new_nodes = __node_split__(pcfg, nodes.pop())
        while not success:
            i += 1
            nodes.append(new_nodes[0])
            success, new_nodes = __node_split__(pcfg, nodes.pop(-i))
        for new_node in new_nodes:
            insertion_index: int = bisect.bisect(nodes, new_node)
            nodes.insert(insertion_index, new_node)

    return nodes


def __holes_of__(pcfg: PCFG, node: Node) -> List[Context]:
    stack = [pcfg.start]
    current = node.program[:]
    while current:
        S = stack.pop()
        P = current.pop(0)
        args = pcfg.rules[S][P][0]
        for arg in args:
            stack.append(arg)
    return stack


def __is_fixing_any_hole__(
    pcfg: PCFG, node: Node, holes: List[Context]
) -> bool:
    current = node.program[:]
    stack = [pcfg.start]
    while current:
        S = stack.pop()
        if S in holes:
            return True
        P = current.pop(0)
        args = pcfg.rules[S][P][0]
        for arg in args:
            stack.append(arg)
    return False


def __are_compatible__(pcfg: PCFG, node1: Node, node2: Node) -> bool:
    """
    Two nodes prefix are compatible if one does not fix a context for the other.
    e.g. a -> b -> map -> *  and c -> b -> map -> +1 -> * are incompatible.

    In both cases map have the same context (bigram context) which is ((predecessor=b, argument=0), depth=2) thus are indistinguishables.
    However in the former all derivations are allowed in this context whereas in the latter +1 must be derived.
    Thus we cannot create a CFG that enables both.
    """
    holes1 = __holes_of__(pcfg, node1)
    if __is_fixing_any_hole__(pcfg, node2, holes1):
        return False
    holes2 = __holes_of__(pcfg, node2)
    return not __is_fixing_any_hole__(pcfg, node1, holes2)


def __all_compatible__(pcfg: PCFG, node: Node, group: List[Node]) -> bool:
    return all(__are_compatible__(pcfg, node, node2) for node2 in group)


def __try_split_node_in_group__(
    pcfg: PCFG, prob_groups: List[List], group_index: int
) -> bool:
    group_a: List[Node] = prob_groups[group_index][1]
    # Sort group by ascending probability
    group_a_bis = sorted(group_a, key=lambda x: x.probability)
    # Try splitting a node until success
    i = 1
    success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
    while not success and i < len(group_a):
        i += 1
        success, new_nodes = __node_split__(pcfg, group_a_bis[-i])
    if i >= len(group_a):
        return False
    # Success, remove old node
    group_a.pop(-i)
    # Add new nodes
    for new_node in new_nodes:
        group_a.append(new_node)
    return True


def __find_swap_for_group__(
    pcfg: PCFG, prob_groups: List[List], group_index: int
) -> Optional[Tuple[int, Optional[int], int]]:
    max_prob: float = prob_groups[-1][1]
    min_prob: float = prob_groups[0][1]
    group_a, prob = prob_groups[group_index]
    best_swap: Optional[Tuple[int, Optional[int], int]] = None
    current_score: float = max_prob / prob

    candidates = (
        list(range(len(prob_groups) - 1, group_index, -1))
        if group_index == 0
        else [len(prob_groups) - 1]
    )

    for i in candidates:
        group_b, prob_b = prob_groups[i]
        for j, node_a in enumerate(group_a):
            pa: float = node_a.probability
            reduced_prob: float = prob - pa
            # Try all swaps
            for k, node_b in enumerate(group_b):
                pb: float = node_b.probability
                if (
                    pb < pa
                    or not __all_compatible__(pcfg, node_a, group_b)
                    or not __all_compatible__(pcfg, node_b, group_a)
                ):
                    continue
                new_mass_b: float = prob_b - pb + pa
                mini = min_prob if group_index > 0 else reduced_prob + pb
                maxi = (
                    max(new_mass_b, prob_groups[-2][1])
                    if j == len(prob_groups) - 1
                    else max_prob
                )
                new_score = maxi / mini
                if new_score < current_score:
                    best_swap = (i, j, k)
                    current_score = new_score
        # Consider taking something from b
        for k, node_b in enumerate(group_b):
            if not __all_compatible__(pcfg, node_b, group_a):
                continue
            pb = node_b.probability
            if prob + pb > max_prob:
                new_score = (prob + pb) / min_prob
            else:
                new_score = max_prob / (prob + pb)
            if new_score < current_score:
                best_swap = (i, None, k)
                current_score = new_score
    return best_swap


def __percolate_down__(prob_groups: List[List], group_index: int) -> None:
    index = group_index
    p = prob_groups[group_index][1]
    while index > 0 and prob_groups[index - 1][1] > p:
        prob_groups[index - 1], prob_groups[index] = (
            prob_groups[index],
            prob_groups[index - 1],
        )
        index -= 1


def __percolate_up__(prob_groups: List[List], group_index: int) -> None:
    index = group_index
    p = prob_groups[group_index][1]
    while index < len(prob_groups) - 2 and prob_groups[index + 1][1] < p:
        prob_groups[index + 1], prob_groups[index] = (
            prob_groups[index],
            prob_groups[index + 1],
        )
        index += 1


def __apply_swap__(
    prob_groups: List[List], group_index: int, swap: Tuple[int, Optional[int], int]
) -> None:
    j, k, l = swap
    # App
    if k:
        node_a = prob_groups[group_index][0].pop(k)
        prob_groups[group_index][1] -= node_a.probability
        prob_groups[j][0].append(node_a)
        prob_groups[j][1] += node_a.probability

    node_b = prob_groups[j][0].pop(l)
    prob_groups[j][1] -= node_b.probability
    prob_groups[group_index][0].append(node_b)
    prob_groups[group_index][1] += node_b.probability

    __percolate_down__(prob_groups, -1)
    __percolate_up__(prob_groups, group_index)


def __split_into_nodes__(
    pcfg: PCFG, splits: int, desired_ratio: float = 2
) -> Tuple[List[List[Node]], float]:
    nodes = __split_nodes_until_quantity_reached__(pcfg, splits)

    # Create groups
    groups: List[List[Node]] = []
    for node in nodes[:splits]:
        groups.append([node])
    for node in nodes[splits:]:
        # Add to first compatible group
        added = False
        for group in groups:
            if __all_compatible__(pcfg, node, group):
                group.append(node)
                added = True
                break
        assert added

    # Improve
    improved = True
    masses: List[float] = [
        np.sum([x.probability for x in group]) for group in groups]
    prob_groups = sorted([[g, p] for g, p in zip(
        groups, masses)], key=lambda x: x[1])  # type: ignore
    ratio: float = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    while improved and ratio > desired_ratio:
        improved = False
        for i in range(splits - 1):
            swap = __find_swap_for_group__(pcfg, prob_groups, i)
            if swap:
                improved = True
                __apply_swap__(prob_groups, i, swap)
                break
        if not improved:
            for i in range(splits - 1, 0, -1):
                improved = __try_split_node_in_group__(pcfg, prob_groups, i)
                if improved:
                    break
        ratio = prob_groups[-1][1] / prob_groups[0][1]  # type: ignore
    return [g for g, _ in prob_groups], ratio  # type: ignore


def split(
    pcfg: PCFG, splits: int, alpha: float = 1.1
) -> Tuple[List[PCFG], float]:
    """
    Currently use exchange split.
    Parameters:
    alpha: the max ratio authorized between the most probable group and the least probable pcfg

    Return:
    a list of PCFG
    the reached threshold
    """
    if splits == 1:
        return [pcfg], 1
    assert alpha > 1, "The desired ratio must be > 1!"
    groups, ratio = __split_into_nodes__(pcfg, splits, alpha)
    return [__pcfg_from__(pcfg, group) for group in groups if len(group) > 0], ratio
