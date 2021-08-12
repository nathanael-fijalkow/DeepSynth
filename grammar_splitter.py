from pcfg import PCFG
from typing import Any, Dict, List, Optional, Tuple
import bisect
import numpy as np
from cons_list import cons_list2list, length, cons_list_split

cons_list = Any
Context = Any
NodeData = Tuple[float, List[Context], cons_list, cons_list]


def __common_prefix__(a: cons_list, b: cons_list) -> cons_list:
    if a == b:
        return a
    possibles = []
    try:
        _, ta = a
        possibles.append(__common_prefix__(ta, b))
    except:
        return None
    try:
        _, tb = b
        possibles.append(__common_prefix__(a, tb))
    except:
        return None
    lentghs = [length(x) for x in possibles]
    if max(lentghs) == lentghs[0]:
        return possibles[0]
    return possibles[1]


def __remove_prefix__(seq: cons_list, prefix: cons_list, len_seq: Optional[int] = None, len_prefix: Optional[int] = None) -> cons_list:
    l_seq = len_seq or length(seq)
    l_prefix = len_prefix or length(prefix)
    if l_seq == l_prefix:
        return None
    return (seq[0], __remove_prefix__(seq[1], prefix, l_seq - 1, l_prefix))


def __pcfg_from__(original_pcfg: PCFG, group: List[NodeData]) -> Tuple[cons_list, PCFG]:
    # print("=" * 60)
    # Find the common prefix to all
    min_prefix: cons_list = group[0][-1]
    for _, _, x, deriv_prefix in group[1:]:
        min_prefix = __common_prefix__(min_prefix, deriv_prefix)
        # print("\t", x)

    # for _, _, x, _ in group:
    #     for _, _ , y, _ in group:
    #         print("x=", x, "y=",y, "compatible=", __are_compatible__(original_pcfg, x, y))
    assert min_prefix is not None
    # Extract the start symbol
    start, min_prefix = min_prefix
    # print("Min_prefix=", (start, min_prefix))
    # print("Max program prefix:", group[0][2])
    # print("Start=", start)
    # Mark all paths that should be filled
    to_fill = []
    rules = {start: {}}
    for _, args, program, prefix in group:
        # Create rules to follow the path
        ctx_path: List = cons_list2list(__remove_prefix__(prefix, (start, min_prefix)))[::-1]
        program_path: List = cons_list2list(__remove_prefix__(program, min_prefix))[::-1]
        for i, P in enumerate(program_path):
            current = start if i == 0 else ctx_path[i - 1]
            if current not in rules:
                rules[current] = {}
            rules[current][P] = original_pcfg.rules[current][P][0], 10 # 10 because in logprobs and probs it doesn't make sense so easy to see an error
            # print("\t", current, "->", P)
        # If there is no further derivation
        if not args:
            continue
        # Next derivation should be filled
        # print("From", program, "args:", args)
        for arg in args:
            to_fill.append(arg)
        
    # At this point rules can generate all partial programs
    # Get the S to normalize by descending depth order
    to_normalise = sorted(list(rules.keys()), key=lambda x: x[-1])
    # print("To normalise:", to_normalise)
    # print("To fill:", [x[1][0] for x in to_fill])

    # Build rules from to_fill
    while to_fill:
        S = to_fill.pop()
        # print("\tFilling:", S)
        rules[S] = {}
        for P in original_pcfg.rules[S]:
            args, w = original_pcfg.rules[S][P]
            rules[S][P] = (args[:], w) # copy list
            for arg in args:
                if arg not in rules:
                    to_fill.append(arg)
    # So far so good works as expected

    # At this point we have all the needed rules
    # However, the probabilites are incorrect
    while to_normalise:
        S = to_normalise.pop()
        # Compute the updated probabilities
        for P in list(rules[S]):
            args, _ = rules[S][P]
            # We have the following equation:
            # (1) w = old_w * remaining_fraction
            old_w = original_pcfg.rules[S][P][1]
            remaining_fraction = 1
            # If there is a next derivation use it to compute the remaining_fraction
            if args:
                N = args[-1]
                remaining_fraction = sum(rules[N][K][1] for K in rules[N])
            # Update according to Equation (1)
            rules[S][P] = args, old_w  * remaining_fraction

        # The updated probabilities may not sum to 1 so we need to normalise them
        # But let PCFG do it with clean=True

    min_depth: int = start[-1]
    program_prefix: cons_list = cons_list_split(group[0][2], length(group[0][2]) - length(min_prefix))[1]
    # print("Program prefix=", program_prefix)

    # Now our min_prefix may be something like MAP, which takes 2 arguments 
    # but the generators need to know that the start symbol is simply not (?, (MAP, 1), 1)
    # that is there must be a (?, (MAP, 0), 1) generated afterwards

    # Compute missing arguments to generate
    derivations = cons_list2list(min_prefix)[::-1]
    program_path = cons_list2list(program_prefix)[::-1]
    stack = []
    for S, P in zip(derivations, program_path):
        argsP, w = original_pcfg.rules[S][P]
        if stack:
            Sp, Pp, n = stack.pop()
            if n > 1:
                stack.append((Sp, Pp, n - 1))
        if len(argsP) > 0:
            stack.append((S, P, len(argsP)))
    # We are missing start so we have to do it manually
    if stack:
        Sp, Pp, n = stack.pop()
        if n > 1:
            stack.append((Sp, Pp, n - 1))
            
    # print("derivations=", derivations)
    # print("program_path=", program_path)
    # print("stack=", stack)
    # Stack contains all the HOLES
    l = length(min_prefix)
    i = -1
    while stack:
        S, P, n = stack.pop()
        start = S
        while True:
            St, Pt = derivations[i], program_path[i]
            i -= 1
            rules[St] = {Pt: (original_pcfg.rules[St][Pt][0], 1.0)}
            if St == S and Pt == P:
                break
        # rules[start] = {P: (original_pcfg.rules[start][P], 1.0)}
    l += i + 1

    # print("start=", start)
    min_depth: int = start[-1]
    program_prefix: cons_list = cons_list_split(
        group[0][2], length(group[0][2]) - l)[1]
    # print("Program prefix=", program_prefix)

    # Ensure rules are depth ordered
    rules = {key: rules[key] for key in sorted(
        list(rules.keys()), key=lambda x: x[-1])}

    # Update max depth
    max_depth: int = original_pcfg.max_program_depth - min_depth
    # print("Symbols=", list(rules.keys()))

    return program_prefix, PCFG(start, rules, max_depth, clean=True)



def __node_split__(pcfg: PCFG, node: NodeData) -> Tuple[bool, List[NodeData]]:
    """
    Split the specified node accordingly.

    Return: success, nodes: 
    - True, list of children nodes
    - False, [node]
    """
    output = []
    prob, derivations, program, deriv_history = node
    # If there is no more derivation then it means this node can't be split
    if len(derivations) == 0:
        return False, [node]
    derivation: Context = derivations.pop()
    for P in pcfg.rules[derivation]:
        args, p_prob = pcfg.rules[derivation][P]
        new_root = (prob * p_prob, derivations + args,
                    (P, program), (derivation, deriv_history))
        output.append(new_root)
    return True, output

def __threshold_split_nodes__(pcfg: PCFG, threshold: float, root: Optional[NodeData] = None) -> List[NodeData]:
    """
    Start from the root node and split most probable node until the most probable node is inferior in logprobs to the threshold in logprobs.
    """
    nodes: List[NodeData] = root or  [(1, [pcfg.start], None, None)]
    # Split nodes until their probability is less than the threshold
    while nodes[-1][0] > threshold:
        i = 1
        success, new_nodes = __node_split__(pcfg, nodes.pop())
        while not success:
            nodes.append(new_nodes[0])
            i += 1
            success, new_nodes = __node_split__(pcfg, nodes.pop(-i))
        for new_node in new_nodes:
            insertion_index: int = bisect.bisect(nodes, new_node)
            nodes.insert(insertion_index, new_node)

    return nodes


def __quantity_split_nodes__(pcfg: PCFG, threshold: int, root: Optional[NodeData] = None) -> List[NodeData]:
    """
    Start from the root node and split most probable node until the threshold number of ndoes is reached.
    """
    nodes: List[NodeData] = root or [(1, [pcfg.start], None, None)]
    while len(nodes) < threshold:
        i = 1
        success, new_nodes = __node_split__(pcfg, nodes.pop())
        while not success:
            i += 1
            success, new_nodes = __node_split__(pcfg, nodes.pop(-i))
        for new_node in new_nodes:
            insertion_index: int = bisect.bisect(nodes, new_node)
            nodes.insert(insertion_index, new_node)

    return nodes


def find_partition(numbers, target, maximum_new_groups):
    """
    7/6 bound on optimal maximum sum partition.
    """
    target_groups = min(maximum_new_groups, int(np.sum(numbers) / target))
    target_groups = max(2, target_groups)
    groups = [[[], 0] for _ in range(target_groups)]
    for i, n in sorted(enumerate(numbers), reverse=True, key=lambda x: x[1]):
        best = 0
        dist = 999999
        for j, (_, somme) in enumerate(groups):
            d = abs((somme + n) - target) - abs(somme - target)
            if d < dist:
                dist = d
                best = j
        groups[best][0].append(i)
        groups[best][1] += n
    return [index for index, _ in groups if index]


def __are_compatible__(pcfg, pa: cons_list, pb: cons_list) -> bool:
    """
    Check if the two prefix program const list are compatible
    i.e. if one of them does not fix a HOLE of the other (with the same context)
    """
    lpa = cons_list2list(pa)
    lpb = cons_list2list(pb)
    stackA = [pcfg.start]
    stackB = [pcfg.start]
    while lpa and lpb:
        Sa = stackA.pop()
        Sb = stackB.pop()

        Pa = lpa.pop()
        Pb = lpb.pop()
        argsA = pcfg.rules[Sa][Pa][0]
        for arg in argsA:
            stackA.append(arg)
        argsB = pcfg.rules[Sb][Pb][0]
        for arg in argsB:
            stackB.append(arg)

    common = None
    for el in stackA:
        if el in stackB:
            if common:
                common.append(el)
            else:
                common = [el]

    if common:
        while lpa:
            Sa = stackA.pop()
            Pa = lpa.pop()
            argsA = pcfg.rules[Sa][Pa][0]
            for arg in argsA:
                stackA.append(arg)

        while lpb:
            Sb = stackB.pop()
            Pb = lpb.pop()
            argsB = pcfg.rules[Sb][Pb][0]
            for arg in argsB:
                stackB.append(arg)

        new_common = None
        for el in stackA:
            if el in stackB:
                if new_common:
                    new_common.append(el)
                else:
                    new_common = [el]
        return common == new_common




    return True


def all_compatible(pcfg, pa, lpb, i):

    for j, pb in enumerate(lpb):
        if i == j:
            continue
        if not __are_compatible__(pcfg, pa, pb):
            return False
    return True
# TODO: If you want to revive those methods you need to make sure the partitions created are compatible
# def threshold_group_split(pcfg: PCFG, splits: int, alpha: float = 2):
#     nodes: List[NodeData] = __threshold_split_nodes__(pcfg, alpha / splits)
#     # Now we may have more splits than necessary
#     # Idea: gather them together such that out of n quantities we can make k groups of equal mass
#     # This is actually the partition problem => NP-hard :/
#     # Instead we'll "simplify" the problem
#     # Only nodes that have the same prefix can be grouped together
#     groups: Dict[Tuple, int] = {}
#     for _, _, _, prefix in nodes:
#         if prefix not in groups:
#             groups[prefix] = 1
#         else:
#             groups[prefix] += 1
#     # Ok keep all that can't be grouped together
#     alone_roots: List[Tuple[float, Context, cons_list, cons_list]] = [
#         x for x in nodes if groups[x[-1]] == 1]

#     # Remap groups to unique id
#     id = 0
#     group_id: Dict[cons_list, int] = {}
#     for key, size in groups.items():
#         if size > 1:
#             group_id[key] = id
#             id += 1
#     # Create the groups
#     del groups
#     groups = [[] for _ in range(id)]
#     group_mass = [[] for _ in range(id)]
#     for el in nodes:
#         id = group_id[el[-1]]
#         groups[id].append(el)
#         group_mass[id].append(el[0])
#     for i, el in enumerate(group_mass):
#         group_mass[i] = np.sum(group_mass[i])

#     # Okay so now we know how much mass there is in each group
#     # What we want to do is break down groups
#     # For each group we have now a partition problem => NP-HARD :/
#     # But actually it's great because we reduced the number of items
#     while len(alone_roots) + len(groups) < splits:
#         largest_group = np.argmax(group_mass)
#         elements = groups.pop(largest_group)
#         group_mass.pop(largest_group)
#         maximum_creatable_groups = splits - len(alone_roots) - len(groups)
#         indices = find_partition([x[0] for x in elements], 1 / splits, maximum_creatable_groups)
#         new_groups = [[x for i, x in enumerate(elements) if i in index] for index in indices]
#         # Add the new groups
#         for group in new_groups:
#             if len(group) == 1:
#                 alone_roots.append(group[0])
#             else:
#                 groups.append(group)
#                 group_mass.append(np.sum([x[0] for x in group]))

#     # Add all the alone roots
#     groups += [[el] for el in alone_roots]
#     group_mass += [el[0] for el in alone_roots]
#     return groups


# def threshold_split(pcfg: PCFG, splits: int, alpha: float = 2):
#     nodes: List[NodeData] = __threshold_split_nodes__(pcfg, alpha / splits)


#     probs = [l for l, _, _, _ in nodes]
#     partitions = find_partition(probs, 1/splits, splits)
#     groups = [[x for i, x in enumerate(nodes) if i in indices]
#               for indices in partitions]
#     return groups


def threshold_exchange_split(pcfg: PCFG, splits: int, alpha: float = 2):
    nodes = __threshold_split_nodes__(pcfg, alpha /splits)
    # Create groups
    groups = []
    for node in nodes[:splits - 1]:
        groups.append([node])
    groups.append(nodes[splits - 1:])

    def __exchange__(groups, masses, a, b):
        group_a = groups[a]
        group_b = groups[b]
        current_score = masses[a] / masses[b]
        best_swap = None

        for i, ela in enumerate(group_a):
            pa = ela[0]
            red_mass_a = masses[a] - pa
            for j, elb in enumerate(group_b):
                pb = elb[0]
                if not all_compatible(pcfg, ela[2], group_b, j) or not all_compatible(pcfg, elb[2], group_a, i):
                    continue
                new_mass_b = masses[b] - pb + pa
                new_score = (red_mass_a + pb) / new_mass_b
                if new_score <= 0:
                    continue
                if new_score < 1:
                    new_score = 1/ new_score
                if new_score < current_score:
                    best_swap = (i, j)
                    current_score = new_score
            # Consider giving out
            if not all_compatible(pcfg, ela[2], group_b, -1):
                continue
            new_score = red_mass_a / (masses[b] + pa)
            if new_score <= 0:
                continue
            if new_score < 1:
                new_score = 1/ new_score
            if new_score < current_score:
                best_swap = (i, None)
                current_score = new_score
        if best_swap is None:
            return False
        i, j = best_swap
        if j is not None:
            group_a[i], group_b[j] = group_b[j], group_a[i]
        else:
            group_b.append(group_a.pop(i))
            
        return True
    # Improve
    improved = True
    masses = [np.sum([x[0] for x in group]) for group in groups]
    while improved:
        best = np.argmax(masses)
        worst = np.argmin(masses)
        improved = __exchange__(groups, masses, best, worst)
        masses[best] = np.sum([x[0] for x in groups[best]])
        masses[worst] = np.sum([x[0] for x in groups[worst]])

    return groups


def exchange_split(pcfg: PCFG, splits: int, alpha: float = 2):
    nodes = __quantity_split_nodes__(pcfg, splits)
    # Create groups
    groups = []
    for node in nodes[:splits - 1]:
        groups.append([node])
    groups.append(nodes[splits - 1:])

    def __try_split__(groups, a):
        group_a = groups[a]
        group_a_bis = sorted(group_a, key=lambda x: x[0])
        i = 1
        success, new_nodes = __node_split__(pcfg, group_a_bis.pop())
        while not success and i < len(group_a):
            i += 1
            success, new_nodes = __node_split__(pcfg, group_a_bis.pop())
        if i > len(group_a):
            return False
        group_a.pop(-i)
        for new_node in new_nodes:
            group_a.append(new_node)

        return True

    def __exchange__(groups, masses, a, b, splits_done=0):
        group_a = groups[a]
        if len(group_a) == 1 and splits_done <= 0:
            return __try_split__(groups, a) and __exchange__(groups, masses, a, b, splits_done+1)
        group_b = groups[b]
        if len(group_b) == 1 and splits_done <= 0:
            return __try_split__(groups, b) and __exchange__(groups, masses, a, b,  splits_done+1)
        current_score = masses[a] / masses[b]
        best_swap = None

        for i, ela in enumerate(group_a):
            pa = ela[0]
            red_mass_a = masses[a] - pa
            for j, elb in enumerate(group_b):
                if not all_compatible(pcfg, ela[2], group_b, j) or not all_compatible(pcfg, elb[2], group_a, i):
                    continue
                pb = elb[0]
                new_mass_b = masses[b] - pb + pa
                new_score = (red_mass_a + pb) / new_mass_b
                if new_score <= 0:
                    continue
                if new_score < 1:
                    new_score = 1 / new_score
                if new_score < current_score:
                    best_swap = (i, j)
                    current_score = new_score
            # Consider giving out
            if not all_compatible(pcfg, ela[2], group_b, -1):
                continue
            new_score = red_mass_a / (masses[b] + pa)
            if new_score <= 0:
                continue
            if new_score < 1:
                new_score = 1 / new_score
            if new_score < current_score:
                best_swap = (i, None)
                current_score = new_score
        if best_swap is None:
            if current_score > alpha:   
                if splits_done > 0:
                    return False       
                return __try_split__(groups, a) and __exchange__(groups, masses, a, b, splits_done+1)
            return False
        i, j = best_swap
        if j is not None:
            group_a[i], group_b[j] = group_b[j], group_a[i]
        else:
            group_b.append(group_a.pop(i))

        return True
    # Improve
    improved = True
    masses = [np.sum([x[0] for x in group]) for group in groups]
    while improved:
        best = np.argmax(masses)
        worst = np.argmin(masses)
        improved = __exchange__(groups, masses, best, worst)
        if improved:
            masses[best] = np.sum([x[0] for x in groups[best]])
            masses[worst] = np.sum([x[0] for x in groups[worst]])

    return groups


def split(pcfg: PCFG, splits: int, **kwargs):
    """
    Currently use exchange split.
    Additional parameter:
    alpha: the max ratio authorized between the most probable group and the least probable

    Return:
    a list of Tuple[prefix program cons_list, PCFG]
    """
    return [__pcfg_from__(pcfg, group) for group in exchange_split(pcfg, splits, **kwargs)]

if __name__ == "__main__":
    import dsl
    from DSL.deepcoder import *
    deepcoder = dsl.DSL(semantics, primitive_types)
    type_request = Arrow(List(INT), List(INT))
    np.random.seed(0)
    deepcoder_CFG = deepcoder.DSL_to_CFG(type_request)

    alpha_threshold = 1
    max_ratio = 1.01

    methods = [
        # ("threshold group partition", lambda pcfg, splits: threshold_group_split(pcfg, splits, alpha_threshold)),
        # ("threshold partition", lambda pcfg, splits: threshold_split(pcfg, splits, alpha_threshold)),
        ("threshold exchange", lambda pcfg, splits: threshold_exchange_split(pcfg, splits, alpha_threshold)),
        ("exchange", lambda pcfg, splits: exchange_split(pcfg, splits, max_ratio)),
        ]
    splits = 20
    print("Parameters:")
    print("\tsplits=", splits)
    print("\talpha_threshold=", alpha_threshold)
    print("\tmax_ratio=", max_ratio)
    import tqdm
    samples = 1
    data = {}
    for name, fun in methods:
        data[name] = []
    for i in tqdm.trange(samples):
        deepcoder_PCFG = deepcoder_CFG.CFG_to_Random_PCFG(alpha=1.0)
        for name, fun in methods:
            groups = fun(deepcoder_PCFG, splits)
            group_mass = [np.sum([l for l, _, _, _ in group]) for group in groups]
            ratio = np.max(group_mass) / np.min(group_mass)
            data[name].append(ratio)
    for name, values in data.items():
        print(name, ":")
        print("\tmean:", np.mean(values))
        print("\tmedian:", np.median(values))
        print("\tmax:", np.max(values))
        print("\tmin:", np.min(values))
        print("\tvariance:", np.var(values))


    for name, fun in methods:
        groups = fun(deepcoder_PCFG, splits)
        [__pcfg_from__(deepcoder_PCFG, group) for group in groups]
