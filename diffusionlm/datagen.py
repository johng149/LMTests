import random
class SyntheticDatasetFSM:

    def __init__(self):
        self.states = ['A', 'B', 'C', 'D', 'E']
        self.initial_state_probabilities = {
            'A': 0.2,
            'B': 0.8
        }
        self.transitions = {
            'A': {'B': 0.7, 'C': 0.3},
            'B': {'C': 0.2, 'D': 0.2, 'E': 0.4, 'B': 0.2},
            'C': {'D': 0.8, 'E': 0.2},
            'D': {'E': 1.0},
            'E': {'A': 0.5, 'B': 0.2, 'E': 0.3}
        }

        self._validate_initial_state_probabilities()
        self._validate_transition_probabilities()

    def _validate_initial_state_probabilities(self):
        if sum(self.initial_state_probabilities.values()) != 1.0:
            raise ValueError("Invalid initial state probabilities")

    def _validate_transition_probabilities(self):
        for state in self.transitions:
            probabilities = self.transitions[state]
            if sum(probabilities.values()) != 1.0:
                raise ValueError("Invalid transition probabilities for state: {}".format(state))
            
    def _sample_initial_state(self):
        return random.choices(
            population=list(self.initial_state_probabilities.keys()),
            weights=list(self.initial_state_probabilities.values()),
            k=1
        )[0]
    
    def _sample_next_state(self, current_state):
        return random.choices(
            population=list(self.transitions[current_state].keys()),
            weights=list(self.transitions[current_state].values()),
            k=1
        )[0]
    
    def generate(self, seq_len):
        current_state = self._sample_initial_state()
        seq = [current_state]
        for _ in range(seq_len - 1):
            current_state = self._sample_next_state(current_state)
            seq.append(current_state)
        return seq

def generate_data(num_samples, max_seq_len, min_seq_len):
    fsm = SyntheticDatasetFSM()
    samples = []
    for _ in range(num_samples):
        seq_len = random.randint(min_seq_len, max_seq_len)
        samples.append(fsm.generate(seq_len))
    return samples