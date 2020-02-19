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

        #for check
        self.check_state_vocab()

    def _build_inversed_condition_map(self,condition_map):
        inversed_condition_map={}
        for type in condition_map:
            instances=condition_map[type]
            for inst in instances:
                inversed_condition_map[inst]=type
        return inversed_condition_map

    def check_state_vocab(self):
        state_without_vocab=set()
        for state in self.states:
            if not state in self.state2vocab:
                state_without_vocab.add(state)
        print('DEBUG: FSA state wihtout vocabs: {}'.format(state_without_vocab))
        assert len(state_without_vocab) <= 0




    def reset_states(self):
        self.cur_state=self.start_state

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
        index=1 if inputs[0] == 'start' else 0
        state_list=[]
        self.reset_states()
        state_list.append(self.cur_state)
        while self.cur_state not in self.end_states:
            if index >= len(inputs):
                break
            self.get_next_state(inputs[index])
            state_list.append(self.cur_state)
            index+=1
        self.reset_states()
        return state_list

    def get_vocab_for_states(self,states):
        vocab_indexes=[]
        for state in states:
            vocab_indexes.append(self.get_vocab_index(state))
        return vocab_indexes

    def get_next_state(self,input):
        next_states=self.transitions.get(self.cur_state)
        '''
        print('1: cur:{}, and input:{}, next_states:{}'.
              format(self.cur_state, input,next_states))
        print('2: cur:{}, and input:{}, next_states:{}, input_type: {}'.
              format(self.cur_state, input,next_states,self.condition_map.get(input,None)))
        '''
        if not input in self.condition_map:
            print(self.condition_map)
            print('input:')
            print(input)
        self.cur_state=next_states.get(self.condition_map[input])
        return self.cur_state

    def get_vocab_index(self,state):
        if not state in self.state2vocab:
            print("VOCAB ERROR: {} has no vocab".format(state))
            exit(-1)
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



