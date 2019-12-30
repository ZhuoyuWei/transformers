import json

class FiniteStateAutomata:
    def __init__(self,states,start_state,end_states,transitions):
        '''
        How to define a grammar or even if-else to build the rule of automata
        '''
        self.states=states
        self.start_state=start_state[0]
        self.end_states=set(end_states)
        self.transitions = self._indexing_transitions(transitions)

        self.cur_state=self.start_state

    def _indexing_transitions(self,transitions):
        transition_dict={}
        for transition in transitions:
            if not transition["from"] in transition_dict:
                transition_dict[transition["from"]]={}
            transition_dict[transition["from"]][transition["condition"]]=transition["to"]
        return transition_dict

    def convert_seq_to_states(self,inputs):
        '''
        give a legal seq, convert it to the seq of states
        call get_next_state
        :return:
        '''
        index=0
        state_list=[]
        while self.cur_state != self.end_states:
            if index >= inputs:
                break
            self.get_next_state(inputs[index])
            state_list.append(self.cur_state)
            index+=1
        return state_list


    def get_next_state(self,input):
        next_states=self.transitions.get(self.cur_state)
        self.cur_state=next_states.get(input)
        return self.cur_state


    @classmethod
    def init_from_config(cls,json_file):
        with open(json_file,'r',encoding='utf-8') as f:
            jobj=json.load(f)
        states=jobj["states"]
        start_states=jobj["start_states"]
        end_states=jobj["end_states"]
        transitions=jobj["transition"]
        return cls(states=states,
                   start_state=start_states,
                   end_states=end_states,
                   transitions=transitions)



