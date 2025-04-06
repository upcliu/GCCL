import json
from data_process_tools import pygment_mul_line, split_variable
import numpy as np
import random
import re


class Data4CopynetV3:
    def __init__(self):
        # initial all variable
        # save msg data as list
        self.msgtext = []
        # save splited msg data as list
        self.msg = []
        # save diff generated code change as list
        self.difftext = []
        # split diff and save as list
        self.difftoken = []
        # split diff variable as diff attribution
        self.diffatt = []
        # + or - mark before a token or a line,
        # when it's length is smaller than diff token, it's marking a line.
        # when they're equal, it's marking a token
        self.diffmark = []
        # dict for both diff token and msg word
        self.word2index = {}
        # if a word can't be generated, set it's genmask to 0
        self.genmask = []
        # if a word can't be copy, set it's copymask to 0
        self.copymask = []
        # save the dict for entity and it's representation
        self.variable = []
        # save the word index of the first word that only appear in msg
        self.difftoken_start = 0
        # save the flag
        self.flag = []
    


    def build(self, jsonfile, all_java, one_block):
        # this function read data in json file and fill msg*, diff*, variable
        # jsonfile is file name of a json file. string
        # all_java means only java code change. boolean
        # one_block means only one diff block. boolean
        with open(jsonfile, 'r', encoding='utf-8') as file:
            data = json.load(file)
        """ data = json.load(open(jsonfile, 'r'),  encoding=('utf-8')) """
        # data = json.loads(open(jsonfile).read)
        pattern = re.compile(r'\w+')    # for splitting msg
        """ for i in data:
            print(i) """ 
        jjjj=0
        for x, i in enumerate(data):
            if x > 1000000:  # x for debug, set value of x to a small num
                break
            diff = i['commit']
            # print(diff)
            # files = diff.count('diff --git')
            # java_files = diff.count('.java') // 4
            # blocks = diff.count('@@') // 2
            """ if all_java and (files < 1 or files != java_files):
                continue
            if one_block and blocks != 1:
                continue """
            # print(diff)
            ls = diff.splitlines()
            java_lines = list()
            diff_marks = list()
            other_file = False
            
            """ java_lines.append('NewBlock')
            diff_marks.append(2)
            print(java_lines) """
            for line in ls:
                if len(line) < 1:
                    continue
                if line.startswith('+++') or line.startswith('---'):
                    if not line.endswith('.java'):
                        other_file = True
                        """ jjjj=jjjj+1
                        print(jjjj) """
                        break
                    continue
                # print(line)
                st = line[0]
                # print(st)
                line = line[1:].strip()
                if st == ' ':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/')or line.startswith('//') :
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(2)
                elif st == '-':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/')or line.startswith('//') :
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(1)
                elif st == '+':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/')or line.startswith('//') :
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(3)
            if other_file:
                continue
            tokenList, varDict = pygment_mul_line(java_lines)
            msg = pattern.findall(i['msg'])
            msg = [i for i in msg if i != '' and not i.isspace()]
            self.msgtext.append(i['msg'])
            self.msg.append(msg)
            

            # print(varDict)
            # print(tokenList)
            """ msg = pattern.findall(i['msgs'])
            msg = [i for i in msg if i != '' and not i.isspace()] """
            """ self.msgtext.append(i['msgs'])
            self.msg.append(msg) """
            self.difftext.append(diff)
            # length of diff token and diff mark aren't equal
            """ self.difftoken.append('<nb>') """
            self.difftoken.append(tokenList)
            # print(diff_marks)
            self.diffmark.append(diff_marks)
            self.variable.append(varDict)
            # print(self.difftoken)
            flag = i['tag']
            self.flag.append(flag)
        self.save_data(11,True, True, True, True, True,False)

    def re_process_diff(self):
        # split variable in diff and save it into diff attribution
        # this function should be invoked after build only once
        diff_tokens, diff_marks, diff_atts = [], [], []
        for i, j, k in zip(self.difftoken, self.diffmark, self.variable):
            diff_token, diff_att = [], []
            for x in i:
                if x in k:
                    diff_att.append(split_variable(x))
                    diff_token.append(x)
                else:
                    diff_att.append([])
                    diff_token.append(x)
            #print(diff_att)
            #
            #print(diff_token)
            diff_mark, diff_token, diff_att = self.mark_token(j, diff_token, diff_att)
           
            # translate line mark into token mark
            diff_tokens.append(diff_token)
            diff_marks.append(diff_mark)
            diff_atts.append(diff_att)
        self.diffmark = diff_marks
        self.difftoken = diff_tokens
        self.diffatt = diff_atts
        self.save_data(11,False, False, True, False, False, False)


    def mark_token(self, marklist, tokenlist, attlist):
        lineNum = 0
        diff_mark = list()
        for i in tokenlist:
            lenght=len(marklist)
            if lineNum >= lenght:
                break
            diff_mark.append(marklist[lineNum])
            if i == '<nl>':
                lineNum += 1
        while lineNum < len(marklist):
            diff_mark.append(marklist[lineNum])
            tokenlist.append('<nl>')
            attlist.append([])
            lineNum += 1
        return diff_mark, tokenlist, attlist

   
   

   

    def save_data(self, version, save_difftext=False, save_msgtext=False, save_diff=False, save_msg=False,
                  save_variable=False, save_word2index=False):
        # you can do it any time, just save all data
        if save_difftext:
            json.dump(self.difftext, open('/home/qustliu/AST/data/difftext.json', 'w'))
        if save_msgtext:
            json.dump(self.msgtext, open('/home/qustliu/AST/data/msgtext.json', 'w'))
        if save_diff:
            json.dump(self.difftoken, open('/home/qustliu/AST/data/difftoken.json', 'w'))
            json.dump(self.diffmark, open('/home/qustliu/AST/data/diffmark.json', 'w'))
            json.dump(self.diffatt, open('/home/qustliu/AST/data/diffatt.json', 'w'))
            json.dump(self.flag, open('/home/qustliu/AST/data/flag.json', 'w'))
        if save_msg:
            json.dump(self.msg, open('/home/qustliu/AST/data/diffmsg.json', 'w'))
        if save_variable:
            json.dump(self.variable, open('/home/qustliu/AST/data/diffvariable.json', 'w'))
        if save_word2index:
            json.dump(self.word2index, open('data4CopynetV3/word2indexV{}.json'.format(version), 'w'))
            json.dump(self.genmask, open('data4CopynetV3/genmaskV{}.json'.format(version), 'w'))
            json.dump(self.copymask, open('data4CopynetV3/copymaskV{}.json'.format(version), 'w'))
            json.dump(self.difftoken_start, open('data4CopynetV3/numV{}.json'.format(version), 'w'))


    def load_data(self, version, load_difftext=True, load_msgtext=True, load_diff=True, load_msg=True,
                  load_variable=True, load_word2index=True):
        # java load data from disk
        if load_difftext:
            self.difftext = json.load(open('/home/qustliu/AST/data/difftext.json'))
        if load_msgtext:
            self.msgtext = json.load(open('data4CopynetV3/msgtextV{}.json'))
        if load_diff:
            self.difftoken = json.load(open('/home/qustliu/AST/data/difftoken.json'))
            self.diffmark = json.load(open('/home/qustliu/AST/data/diffmark.json'))
            self.diffatt = json.load(open('/home/qustliu/AST/data/diffatt.json'))
        if load_msg:
            self.msg = json.load(open('data4CopynetV3/msgV{}.json'.format(version)))
        if load_variable:
            self.variable = json.load(open('data4CopynetV3/variableV{}.json'.format(version)))
        if load_word2index:
            self.word2index = json.load(open('data4CopynetV3/word2indexV{}.json'.format(version)))
            self.genmask = json.load(open('data4CopynetV3/genmaskV{}.json'.format(version)))
            self.copymask = json.load(open('data4CopynetV3/copymaskV{}.json'.format(version)))
            self.difftoken_start = int(json.load(open('data4CopynetV3/numV{}.json'.format(version))))
        # print(self.difftoken)
        # print(self.diffmark)
    def filter(self):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        difftoken = json.load(open('/home/qustliu/AST/data/difftoken.json'))
        diffmark = json.load(open('/home/qustliu/AST/data/diffmark.json'))
        for i, j in zip(difftoken, diffmark):
            """ for i, j, k, l, m, n, o in zip(self.difftoken, self.diffmark, self.msg, self.variable, self.difftext,
                                       self.msgtext, self.diffatt): """
            #print(i)
            #print(j)
            diff = []
            for idx, d in enumerate(i):
                if j[0]!=2:
                    if j[0] == 1:
                        diff.append('-')
                        
                    else:
                        diff.append('+')
                        
                if (d == '<nb>' or d == '<nl>') and idx + 1 < len(j) and j[idx + 1] != 2:
                    diff.append(d)
                    if j[idx + 1] == 1:
                        diff.append('-')
                    else:
                        diff.append('+')
                else:
                    diff.append(d)

                na.append(i), nb.append(j)
            # print(na,nb)
        self.difftoken, self.diffmark = na, nb
       
        

if __name__ == '__main__':
    dataset = Data4CopynetV3()
    dataset.build('/home/qustliu/AST/data/updated_data.json', True, False)
    # dataset.load_data(1, True, False, True, False, True, False)
    # dataset.filter()
    
    dataset.re_process_diff()
    print('done')