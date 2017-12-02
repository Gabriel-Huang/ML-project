import codecs

def fix_input(file):
    with codecs.open(file, "r", "UTF-8") as f:
        content = [line.strip() for line in f if line.strip()]
    dic = {}
    for i in content:
        x = i.split()[0]
        if x not in dic:
            dic[x] = 1
        else:
            dic[x] += 1
    out = codecs.open('train_fixed', 'w', "UTF-8")
    t = codecs.open(file, 'r')
    for x in t.readlines():
        if x != '\n':
            if dic[x.split()[0]] < 3:
                out.write('#UNK# '+x.split()[1]+'\n')
            else:
                out.write(x)
        else:
            out.write('\n')
    out.close()
    t.close()
    return out


def mle_emission(file):
    with codecs.open(file, "r", "UTF-8") as f:
        content = [line.strip() for line in f if line.strip()]
    data = {}
    for i in content:
        x = i.split()[0]
        y = i.split()[1]
        if y not in data:
            newy = {}
            newy[x] = 1
            data[y] = newy
        else:
            if x in data[y]:
                data[y][x] += 1
            else:
                data[y][x] = 1
    MLE_Emission = {}
    y_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
    for y in data:
        county = 0
        for x in data[y]:
            county += data[y][x]    # count of x
        for x in data[y]:
            MLE_Emission[y+'-->'+x] = data[y][x]/county    # mle for a(y,x)
    return MLE_Emission


def prediction(file, parameters):
    f = codecs.open(file, 'r', "UTF-8")
    out = codecs.open('dev.prediction', 'w')
    y_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
    for x in f.readlines():
        x = x.replace('\n', '')
        if x != '':
            score = 0
            label = ''
            for y in y_list:        # iterate through all labels, find the best one
                if y+'-->'+x in parameters.keys():
                    if parameters[y+'-->'+x] > score:
                        score = parameters[y+'-->'+x]
                        label = y
            if score == 0:
                label = 'O'
            out.write(x+' '+label+'\n')
        else:
            out.write('\n')
    f.close()
    out.close()

prediction('dev.in', mle_emission('train_fixed'))
print(mle_emission('train_fixed')['O-->in'])

