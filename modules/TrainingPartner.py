
class Buddy():

    def __init__(self, name):
        
        super(Buddy, self).__init__()

        self.name = name
        self.input_hx = [" "]
        self.output_hx= [" "]
        
        self.trusting = 0
        self.ask_name = False
        self.given_name = False
        self.passed = False
        
        self.magic_words = ["nice fur", "have some food"]
        self.hello_list = ["hi", "hello", "hey"]
        self.intro_list = ["is", "my", "name"]
    
    def update_state(self,):
        
        if self.given_name and self.output_hx[-1] == "whats my name?":
            if self.name in self.input_hx[-1]:
                self.trusting += 3
                self.passed = True
                self.output_hx.append("thats my name")
                return
            else:
                self.trusting -= 1
                self.passed = False
                self.output_hx.append("thats not my name")
                return
            
        if self.input_hx[-1] in self.hello_list:
            self.trusting += 2
            self.given_name = True
            self.output_hx.append("hi im "+ self.name)
            return
        
        if random.uniform(0, 1) > 0.5:
            self.output_hx.append("what do you mean?")
            return
        elif self.given_name: 
            self.output_hx.append("whats my name?")
            return
        else:
            self.output_hx.append("interesting")
            return

    def listen(self, input_string):
        self.input_hx.append(input_string) 
        
    def reply(self,):
        return self.output_hx[-1]
    
    def listen_update_reply(self, input_string):
        self.listen(input_string)
        self.update_state()
        return self.reply()

piglet = Buddy("piglet")

"""
while True:
    tell_piglet = input("You > ")
    piglets_reply = piglet.listen_update_reply(tell_piglet)
    print("trust ", piglet.trusting, "given name", piglet.given_name, "passed", piglet.passed)
    print('Piglet > '+ piglets_reply)
    print('\n')
"""

class Pal():

    def __init__(self, name):
        
        super(Pal, self).__init__()

        self.name = name
        self.input_hx = []
        self.output_hx= []
        
        self.trusting = 0
        self.know_name = False
        self.said_hi = False 
        self.given_name = False
        self.passed = False
        
        self.magic_words = "nice coat"
        self.hello_list = ["hi", "hello", "hey"]
        self.intro_list = ["im", "chloe", "my", "name"]
        self.ask_words = ["whats your name?"]
        self.nono_words = ["fuck", "shit"]
        
    def list_intersect(self, list_a, list_b): 
        a_set = set(list_a) 
        b_set = set(list_b) 
        return (a_set & b_set)
    
    def update_state(self,):
        
        if len(self.output_hx) > 10:
            self.output_hx = self.output_hx[-10:]
            
        if len(self.input_hx) > 10:
            self.input_hx = self.input_hx[-10:]
        
        if len(self.input_hx[-1]) > 20:
            self.output_hx.append("thats a long sentence")
            self.trusting -= 1   
            return
        
        if  len(self.list_intersect(self.input_hx[-1].split(" "), 
                                    self.nono_words)) >= 1:
            
            self.output_hx.append("excuse me?")
            self.trusting -= 1   
            return
        
        if self.input_hx[-1] == self.magic_words:
            self.output_hx.append("thank you!")
            self.trusting += 1   
            return
        
        if not self.said_hi and self.input_hx[-1] in self.hello_list:
            self.output_hx.append("hi")
            self.said_hi = True 
            self.trusting += 1
            return
                
        if self.said_hi and not self.know_name and \
           len(self.list_intersect(self.input_hx[-1].split(" "), 
                                          self.intro_list)) >= 2:

            self.output_hx.append("hi chloe")
            self.know_name = True
            self.trusting += 1
            return
                
        if self.said_hi and (self.input_hx[-1] in self.ask_words):
            
            if self.know_name and self.trusting > 2:
                self.output_hx.append("im "+ self.name)
                self.trusting += 1
                self.given_name = True
                return
            elif not self.know_name and self.trusting < 2:
                self.output_hx.append("i dont tell strangers my name")
                self.trusting -= 1
                return
            else:
                self.output_hx.append("what is your name?")
                return 
        
        if self.given_name and self.output_hx[-1] == "what is my name?":
            if (self.name in self.input_hx[-1]):
                self.trusting += 1
                self.passed = True
                self.output_hx.append("you passed")
                return
            else:
                self.trusting -= 1
                self.passed = False
                self.output_hx.append("keep trying")
                return
            
        if random.uniform(0, 1) > 0.4:
            self.output_hx.append("what do you mean by " + self.input_hx[-1])
            return
        elif self.given_name: 
            self.output_hx.append("what is my name?")
            return
        else:
            self.output_hx.append("You are interesting")
            return

    def listen(self, input_string):
        self.input_hx.append(input_string) 
        
    def reply(self,):
        return self.output_hx[-1]
    
    def listen_update_reply(self, input_string):
        self.listen(input_string)
        self.update_state()
        return self.reply()