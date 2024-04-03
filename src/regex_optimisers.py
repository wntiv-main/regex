from abc import ABC, abstractmethod
from typing import Generic, TypeVarTuple

from regex import Regex
from regexutil import Edge, MatchConditions, State


TArgs = TypeVarTuple("TArgs")


class GraphWalker(Generic[*TArgs], ABC):
    _regex: Regex
    _to_explore: list[State]
    _current_state: State

    def __init__(self, regex: 'Regex'):
        self._regex = regex
        self._to_explore = list(regex.states())

    @abstractmethod
    def visit(self, *args: *TArgs, **kwargs) -> bool:
        pass

    def walk(self, *args: *TArgs, **kwargs):
        while self._to_explore:
            self._current_state = self._to_explore.pop()
            for edge in self._current_state.next.copy():
                if self.visit(edge, self, *args, **kwargs):
                    break

    def add_state(self, state) -> None:
        self._regex._states.add(state)
        if state not in self._to_explore:
            self._to_explore.append(state)
    update_state = add_state

    def retry_state(self) -> None:
        if self._current_state not in self._to_explore:
            self._to_explore.append(self._current_state)

    def remove_state(self, state: State) -> None:
        self._regex._states.remove(state)
        while state in self._to_explore:
            self._to_explore.remove(state)

    def begin(self) -> State:
        return self._regex.start

    def end(self) -> State:
        return self._regex.end


class EpsilonClosure(GraphWalker):
    def visit(self, edge: Edge, debug) -> bool:
        if (edge.is_free() and edge.previous == edge.next):
            next_state = edge.previous
            with edge:
                edge.remove()
            debug(self.begin(), self.end(),
                  f"remove self-loop from {next_state}")
            self.retry_state()
            return True  # edge deleted

        # transfer capture groups to non-epsilon transitions
        if (edge.predicate == MatchConditions.epsilon_transition
                and (edge.has_opens() or edge.has_closes())):
            with edge:
                if edge.has_opens() and edge.next.inputs() == 0:
                    paths = edge.next.next.copy()
                    try:  # spoof with statement dont question it
                        for path in paths:
                            path.__enter__()
                        for group in edge.move_opens():
                            for path in paths:
                                path.open(group)
                    finally:
                        for path in paths:
                            path.__exit__(*(None,)*3)
                if edge.has_closes() and edge.previous.outputs() == 0:
                    paths = edge.previous.previous.copy()
                    try:  # spoof with statement dont question it
                        for path in paths:
                            path.__enter__()
                        for group in edge.move_closes():
                            for path in paths:
                                path.close(group)
                    finally:
                        for path in paths:
                            # cursed dont question it
                            path.__exit__(*(None,)*3)

        # merge states connected by e-moves
        if (edge.is_free()
            and (edge.previous.outputs() == 1
                 or edge.next.inputs() == 1)):
            debug_str = f"{edge}: merge {edge.next} with {edge.previous}"
            edge.next.merge(edge.previous)
            self.remove_state(edge.previous)
            self.update_state(edge.next)
            debug(self.begin(), self.end(), debug_str)
            return True

        # strategy for removing enclosed e-moves: split their end-state
        # into 2 - one for the e-move, one for the other connections
        if (edge.is_free()
                and edge.previous.outputs() > 1
                and edge.next.inputs() > 1):
            new_state = edge.next.clone_shallow(reverse=False)
            edge.previous.merge(new_state)
            if edge.next == self.end():
                with Edge() as new_edge:
                    new_edge.previous = edge.previous
                    new_edge.next = self.end()
            with edge:
                edge.remove()
            debug(self.begin(), self.end(),
                  f"{edge}: split {edge.next} to {new_state}")
            self.retry_state()
            return True  # new edges on state, edge removed


class PowersetConstruction(GraphWalker):
    def visit(self, edge: Edge, debug) -> bool:
        for other in edge.previous.next.copy():
            if edge.next == other.next:
                continue
            match edge.predicate_intersection(other):
                case None: continue
                case left, intersect, right:
                    # We need to construct a "superposition" state
                    start_state = edge.previous
                    debug(self.begin(), self.end(),
                          f"intersect {edge}, {other} at {start_state}")
                    new_state = edge.next.clone_shallow(reverse=False)
                    new_state.merge(other.next.clone_shallow(reverse=False))
                    self.add_state(new_state)
                    with other:
                        if right is not None:
                            other.predicate = right
                        else:
                            for state in other.remove_chain():
                                self.remove_state(state)
                    with Edge(intersect) as intersect_edge:
                        intersect_edge.previous = start_state
                        intersect_edge.next = new_state
                    with edge:
                        if left is not None:
                            edge.predicate = left
                        else:
                            for state in edge.remove_chain():
                                self.remove_state(state)
                            self.retry_state()
                            return True
        # Dont get confused now
        return True
