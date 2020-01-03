import json

class FiniteStateAutomata:
    def __init__(self,states,start_state,end_states,transitions,condition_map):
        '''
        How to define a grammar or even if-else to build the rule of automata
        '''
        self.states=states
        self.start_state=start_state[0]
        self.end_states=set(end_states)
        self.transitions, self.state2vocab = self._indexing_transitions(transitions)

        self.cur_state=self.start_state

        self.condition_map=self._build_inversed_condition_map(condition_map)

    def _build_inversed_condition_map(self,condition_map):
        inversed_condition_map={}
        for type in condition_map:
            instances=condition_map[type]
            for inst in instances:
                inversed_condition_map[inst]=type
        return inversed_condition_map



    def _indexing_transitions(self,transitions):
        transition_dict={}
        state2vocab={}
        for transition in transitions:
            if not transition["from"] in transition_dict:
                transition_dict[transition["from"]]={}
            transition_dict[transition["from"]][transition["condition"]]=transition["to"]
            state2vocab[transition["from"]]=transition["vocab"]

        print('DEBUG transition_dict')
        print(transition_dict)
        print('DEBUG state2vocab')
        print(state2vocab)
        return transition_dict,state2vocab

    def convert_seq_to_states(self,inputs):
        '''
        give a legal seq, convert it to the seq of states
        call get_next_state
        :return:
        '''
        index=0
        state_list=[]
        while self.cur_state != self.end_states:
            if index >= len(inputs):
                break
            self.get_next_state(inputs[index])
            state_list.append(self.cur_state)
            index+=1
        return state_list

    def get_vocab_for_states(self,states):
        vocab_indexes=[]
        for state in states:
            vocab_indexes.append(self.get_vocab_index(state))

    def get_next_state(self,input):
        print('cur:{}'.format(self.cur_state))
        next_states=self.transitions.get(self.cur_state)
        self.cur_state=next_states.get(self.condition_map[input])
        return self.cur_state

    def get_vocab_index(self,state):
        return self.state2vocab.get(state,-1)


    @classmethod
    def init_from_config(cls,json_file):
        with open(json_file,'r',encoding='utf-8') as f:
            jobj=json.load(f)
        states=jobj["states"]
        start_states=jobj["start_states"]
        end_states=jobj["end_states"]
        transitions=jobj["transitions"]
        condition_map=jobj["condition_map"]
        return cls(states=states,
                   start_state=start_states,
                   end_states=end_states,
                   transitions=transitions,
                   condition_map=condition_map)



