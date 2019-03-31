# -*- coding: utf-8 -*-

import copy

import tensorflow as tf

from sntp.base import ProofState, NTPParams
from sntp.util import is_variable, is_tensor
from sntp.models.util.masking import create_mask
from sntp.models.util.kmax import k_max
from sntp.index import BaseIndexManager

from sntp.models.util.tfutil import tile_left, naive_top_k

from typing import List, Optional, Union, Any


def unify(atom: List[Union[tf.Tensor, str]],
          goal: List[Union[tf.Tensor, str]],
          proof_state: ProofState,
          ntp_params: NTPParams,
          is_fact: bool = False,
          indices: Optional[tf.Tensor] = None) -> ProofState:

    # symbol-wise unify and min-pooling
    substitutions = copy.copy(proof_state.substitutions)
    scores = proof_state.scores

    if indices is None:
        indices = naive_top_k(atom[0], scores.get_shape())

    f_k = indices.shape[0]

    initial_scores_shp = scores.get_shape()
    goal = [tile_left(elem, initial_scores_shp) for elem in goal]

    atom_tensors_lst = []
    goal_tensors_lst = []

    for atom_index, (atom_elem, goal_elem) in enumerate(zip(atom, goal)):
        if is_variable(atom_elem):
            if atom_elem not in substitutions:
                substitutions.update({atom_elem: goal_elem})
            continue
        elif is_variable(goal_elem):
            if is_tensor(atom_elem):
                atom_shp = atom_elem.get_shape()
                scores_shp = scores.get_shape()

                embedding_size = atom_shp[-1]
                substitution_shp = scores_shp.concatenate([embedding_size])

                f_atom_elem = tf.gather(atom_elem, tf.reshape(indices, [-1]))
                atom_elem = tf.reshape(f_atom_elem, substitution_shp)

            if goal_elem not in substitutions:
                substitutions.update({goal_elem: atom_elem})
            continue
        elif is_tensor(atom_elem) and is_tensor(goal_elem):
            atom_tensors_lst += [atom_elem]
            goal_tensors_lst += [goal_elem]

        #NOTE(zhengyao): I don't understand why concat them, cancel it for now.
        #atom_elem = tf.concat(atom_tensors_lst, axis=-1)
        #goal_elem = tf.concat(goal_tensors_lst, axis=-1)

        goal_elem_shp = goal_elem.get_shape()
        embedding_size = goal_elem_shp[-1]

        # Replicate each sub-goal by the number of facts it will be unified with
        f_goal_elem = tf.reshape(goal_elem, [-1, 1, embedding_size])
        f_goal_elem = tf.tile(f_goal_elem, [1, f_k, 1])
        f_goal_elem = tf.reshape(f_goal_elem, [-1, embedding_size])

        # Move the "most relevant fact dimension per sub-goal" dimension from first to last
        f_indices = tf.transpose(indices, list(range(1, len(indices.shape))) + [0])

        # For each sub-goal, lookup the most relevant facts
        f_new_atom_elem = tf.gather(atom_elem, tf.reshape(f_indices, [-1]))

        # Compute the kernel between each (repeated) sub-goal and its most relevant facts
        f_values = ntp_params.kernel(f_new_atom_elem, f_goal_elem)

        # New shape that similarities should acquire (i.e. [k, g1, .., gn])
        f_scatter_shp = tf.TensorShape(f_k).concatenate(indices.shape[1:])

        # Here similarities have shape [g1 .. gn, k]
        f_values = tf.reshape(f_values, [-1, f_k])

        # Transpose and move the k dimension such that we have [k, g1 .. gn]
        f_values = tf.transpose(f_values, (1, 0))

        # Reshape similarity values
        similarities = tf.reshape(f_values, f_scatter_shp)

        # Reshape the kernel accordingly
        similarities_shp = similarities.get_shape()

        scores_shp = goal_elem.get_shape()[:-1]
        k_shp = tf.TensorShape([f_k])

        target_shp = k_shp.concatenate(initial_scores_shp)

        if similarities_shp != target_shp:
            nb_similarities = tf.size(similarities)

            nb_targets = tf.reduce_prod(target_shp)
            nb_goals = tf.reduce_prod(scores_shp)

            similarities = tf.reshape(similarities, [-1, 1, nb_goals])
            similarities = tf.tile(similarities, [1, nb_targets // nb_similarities, 1])

        similarities = tf.reshape(similarities, target_shp)

        if ntp_params.mask_indices is not None and is_fact:
            # Mask away the similarities to facts that correspond to goals (used for the LOO loss)
            mask_indices = ntp_params.mask_indices
            mask = create_mask(mask_indices=mask_indices, mask_shape=target_shp, indices=indices)

            if mask is not None:
                similarities *= mask

        similarities_shp = similarities.get_shape()
        scores_shp = scores.get_shape()

        if similarities_shp != scores_shp:
            new_scores_shp = tf.TensorShape([1]).concatenate(scores_shp)
            scores = tf.reshape(scores, new_scores_shp)

        scores = tf.minimum(similarities, scores)

    proof_state = ProofState(substitutions=substitutions,
                             scores=scores)
    return proof_state


def substitute(atom: List[Union[tf.Tensor, str]],
               proof_state: ProofState) -> List[Union[tf.Tensor, str]]:
    """
    Implements the SUBSTITUTION method for an atom, given the proof state.

    This is done by traversing through the atom and replacing symbols
    according to the substitution.

    Example:
        atom: [GE, X, Y]
        proof_state:
            scores: [RG
            substitution: {X/GE}

    The result is:
        atom: [RGE, RGE, Y]

    :param atom: Atom.
    :param proof_state: Proof state.
    :return: New atom, matching the proof scores.
    """
    scores_shp = proof_state.scores.get_shape()

    def _process(atom_elem):
        # if atom element is a variable, replace it as specified by the substitution
        res = proof_state.substitutions.get(atom_elem, atom_elem) if is_variable(atom_elem) else atom_elem
        return tile_left(res, scores_shp)
    new_atom = [_process(atom_elem) for atom_elem in atom]
    return new_atom


def neural_and(neural_kb: List[List[List[Union[tf.Tensor, str]]]],
               goals: List[List[Union[tf.Tensor, str]]],
               proof_state: ProofState,
               ntp_params: NTPParams,
               depth: int) -> List[ProofState]:
    """
    Implements the neural AND operator.

    The neural AND operator has the following definition:

    1) AND(_, _, FAIL) = FAIL
    2) AND(_, 0, _) = FAIL
    3) AND([], _, S) = S
    4) AND(g : G, d, S) = [ S'' | S'' \in AND(G, d, S')
                               for S' \in OR(SUBSTITUTE(G, S_psi), d - 1, S) ]

    assume the list of atoms g : G encodes a rule, where g is the head (e.g. [RE, X, Y])
    and G is the body (e.g. [RE, X, Z], [RE, Z, Y]).

    This method proceeds as follows:
    - First it replaces variables in "head" using the current substitution set.
    - Then it calls the OR operator on the new atom.

    Then AND operator is then called recursively on the body G of the rule.

    :param neural_kb: Neural Knowledge Base.
    :param goals: List of atoms.
    :param proof_state: Proof state.
    :param ntp_params: NTP Parameters
    :param depth: Current depth in the proof tree (increased by one when calling neural_or)
    :return: List of proof states.
    """
    # 1) (upstream unification failed) and 2) (depth == max_depth)
    proof_states = []

    if len(goals) == 0:  # 3)
        proof_states = [proof_state]

    elif depth < ntp_params.max_depth:  # 4)
        goal, sub_goals = goals[0], goals[1:]

        new_goal = substitute(goal, proof_state)

        or_proof_states = neural_or(neural_kb=neural_kb,
                                    goals=new_goal,
                                    proof_state=proof_state,
                                    ntp_params=ntp_params,
                                    depth=depth + 1)

        for i, or_proof_state in enumerate(or_proof_states):
            proof_states += neural_and(neural_kb=neural_kb,
                                       goals=sub_goals,
                                       proof_state=or_proof_state,
                                       ntp_params=ntp_params,
                                       depth=depth)

    return proof_states


def top_k(index_manager: BaseIndexManager,
          index: Any,
          atoms: List[Union[tf.Tensor, str]],
          goals: List[Union[tf.Tensor, str]],
          goal_shape: tf.TensorShape,
          k: Optional[int] = 10):
    embedding_size = goal_shape[-1]

    def reshape(tensor: Union[tf.Tensor, str]):
        return tf.reshape(tensor, [-1, embedding_size]) if is_tensor(tensor) else tensor

    new_goals = [reshape(ge) for ge in goals]

    ground_goals = [ge for fe, ge in zip(atoms, new_goals) if is_tensor(fe) and is_tensor(ge)]
    max_dim = max([gg.get_shape()[0] for gg in ground_goals])
    ground_goals = [tf.tile(goal, [max_dim // goal.get_shape()[0], 1]) for goal in ground_goals]

    # [G, 3 E], or e.g. [K, 2 3] if facts or goals contains a variable
    goals_2d = tf.concat(ground_goals, axis=1)

    # Facts in 'facts_2d' most relevant to the query in 'goals_2d'
    goal_2d_np = goals_2d.numpy()
    atom_indices = index_manager.query(index, goal_2d_np, k=k)

    actual_k = atom_indices.shape[-1]
    new_shp = tf.TensorShape(actual_k).concatenate(goal_shape[:-1])
    atom_indices = tf.reshape(tf.transpose(atom_indices), new_shp)
    return tf.cast(atom_indices, tf.int32)


def neural_or(neural_kb: List[List[List[Union[tf.Tensor, str]]]],
              goals: List[Union[tf.Tensor, str]],
              proof_state: ProofState,
              ntp_params: NTPParams,
              depth: int = 0) -> List[ProofState]:
    """
    Implements the neural OR operator.

    It is defined as follows:

    OR(G, d, S) = [ S' | S' \in AND(HEAD, d, UNIFY(HEAD, GOAL, S))
                         for HEAD <- BODY in KB ]

    Assume we have a goal of shape [GE, GE, GE],
    and a rule such as [[RE, X, Y], [RE, X, Z], [RE, Z, Y]].

    This method iterates through all rules (note - facts are just rules with an empty body),
    and unifies the goal (e.g. [GE, GE, GE]) with the head of the rule (e.g. [RE, X, Y]).

    The result of unification is a [RG] tensor of proof scores, and a new set of substitutions
    compatible with the proof scores, i.e. X/RGE and Y/RGE.

    Then, the body of the rule (if present) is reshaped so to match the new proof scores [RG].
    For instance, if the body was [[RE, X, Z], [RE, Z, Y]], the RE tensors are reshaped to GE.

    :param neural_kb: Neural Knowledge Base.
    :param goals: Atom, e.g. [GE, GE, GE].
    :param proof_state: Proof state.
    :param ntp_params: NTP Parameters.
    :param depth: Current depth in the proof tree [default: 0].
    :return: List of proof states.
    """
    if proof_state.scores is None:
        initial_scores = tf.ones(shape=goals[0].get_shape()[:-1], dtype=goals[0].dtype)
        proof_state = ProofState(scores=initial_scores,
                                 substitutions=proof_state.substitutions)

    scores_shp = proof_state.scores.get_shape()
    embedding_size = goals[0].get_shape()[-1]

    goals = [tile_left(elem, scores_shp) for elem in goals]
    goal_shp = scores_shp.concatenate([embedding_size])

    proof_states = []

    for rule_index, rule in enumerate(neural_kb):
        # Assume we unify with a rule, e.g. [[RE X Y], [RE Y X]]
        heads, bodies = rule[0], rule[1:]

        is_fact = len(bodies) == 0

        k = ntp_params.k_facts if is_fact else ntp_params.k_rules

        rule_vars = {e for atom in rule for e in atom if is_variable(e)}
        applied_before = bool(proof_state.substitutions.keys() & rule_vars)

        # In case we reached the maximum recursion depth, do not proceed. Avoids cycles
        if (depth < ntp_params.max_depth or is_fact) and not applied_before:
            index_store = ntp_params.index_store

            if index_store is not None and k is not None:
                index = index_store.get(atoms=heads, goals=goals, position=rule_index)
                indices = top_k(index_manager=index_store.index_manager, index=index,
                                    atoms=heads, goals=goals, goal_shape=goal_shp, k=k)
            else:
                indices = naive_top_k(heads[0], proof_state.scores.get_shape())

            # Unify the goal, e.g. [GE GE GE], with the head of the rule [RE X Y]
            new_proof_state = unify(atom=heads, goal=goals,
                                    proof_state=proof_state, ntp_params=ntp_params,
                                    is_fact=is_fact, indices=indices)

            # Differentiable k-max
            # if k-max-ing is on, and we're processing facts (empty body)
            if ntp_params.k_max and is_fact:
                # Check whether k is < than the number of facts
                if ntp_params.k_max < new_proof_state.scores.get_shape()[0]:
                    new_proof_state = k_max(goals, new_proof_state, k=ntp_params.k_max)

            # The new proof state will be of shape [RG]
            # We now need the body [RE Y X] to match the new proof state as well
            scores_shp = new_proof_state.scores.get_shape()

            # Reshape the rest of the body, so it matches the shape of the head,
            # the current substitutions, and the proof score.
            f_indices = tf.transpose(indices, list(range(1, len(indices.shape))) + [0])

            def normalize_atom_elem(_atom_elem):
                res = _atom_elem
                if is_tensor(res):
                    f_atom_elem = tf.gather(_atom_elem, tf.reshape(f_indices, [-1]))
                    f_atom_shp = scores_shp.concatenate([embedding_size])
                    res = tf.reshape(f_atom_elem, f_atom_shp)
                return res

            def normalize_atom(_atom):
                return [normalize_atom_elem(elem) for elem in _atom]

            new_bodies = [normalize_atom(body_atom) for body_atom in bodies]

            new_body_indices = []
            for atom in new_bodies:
                atom_indices = []
                # Enumeration starts at 1 because the atom indexed at 0 is the one in the head of the rule
                for atom_idx, atom_elem in enumerate(atom, 1):
                    sym_atom_indices = atom_elem

                    atom_indices += [sym_atom_indices]
                new_body_indices += [atom_indices]

            body_proof_states = neural_and(neural_kb=neural_kb,
                                           goals=new_bodies,
                                           proof_state=new_proof_state,
                                           ntp_params=ntp_params,
                                           depth=depth)

            if body_proof_states:
                proof_states += body_proof_states
    return proof_states
