import sys
import math

#Alex Baraban, Homework 2, 09/18/22

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    X = X.fromkeys(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'], 0)
    with open (filename,encoding='utf-8') as f:
        file = f.read().upper()
        for i in range(len(file)):
            if(file[i] in X):
                X[file[i]] += 1
    return X

X = shred('letter.txt')
keys = list(X.keys())
values = list(X.values())
print("Q1")
for i in range(len(keys)):
    print(keys[i] + ' ' + str(values[i]))
    
PEnglish = .6
PSpanish = .4
PEnglishLetter = dict()
PSpanishLetter = dict()
with open('e.txt',encoding='utf-8') as f:
    for line in f:
        key,value = line.split()
        PEnglishLetter[key] = value
with open('s.txt',encoding='utf-8') as f:
    for line in f:
        key,value = line.split()
        PSpanishLetter[key] = value

LogProbSpanishGivenX = math.log(PSpanish)
for key in PSpanishLetter:
    LogProbSpanishGivenX += math.log(float(PSpanishLetter[key])) * float(X[key])
    
LogProbEnglishGivenX = math.log(PEnglish)
for key in PEnglishLetter:   
    LogProbEnglishGivenX += math.log(float(PEnglishLetter[key])) * float(X[key])

if LogProbSpanishGivenX - LogProbEnglishGivenX >= 100:
    NormProbEnglishGivenX = 0
elif LogProbSpanishGivenX - LogProbEnglishGivenX <= -100:
    NormProbEnglishGivenX = 1
else:
    NormProbEnglishGivenX = (1 / (1 + math.exp(LogProbSpanishGivenX - LogProbEnglishGivenX)))

NormProbSpanishGivenX = 1 - NormProbEnglishGivenX

print("Q2")

print("%.4f" %(float(X["A"]) * math.log(float(PEnglishLetter["A"]))))
print("%.4f" %(float(X["A"]) * math.log(float(PSpanishLetter["A"]))))

print("Q3")
print("%.4f" %(LogProbEnglishGivenX))
print("%.4f" %(LogProbSpanishGivenX))

print("Q4")
print("%.4f" %(NormProbEnglishGivenX))