def fix_input(file):
    with open(file) as f:
        content = [line.strip() for line in f if line.strip()]
    dic = {}
    for i in content:
        x = i.split()[0]
        if x not in dic:
            dic[x] = 1
        else:
            dic[x] += 1
    out = open('train_fixed', 'w')
    t = open(file, 'r')
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
    with open(file) as f:
        content = [line.strip() for line in f if line.strip()]
    data = {}       # 2d dictionary: {x1:{y1:count(x1y1), y2:count(x1y2).....}, x2:{..}...}
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
    for x in data:
        county = 0
        for x in data[y]:
            county += data[y][x]    # count of x
        for x in data[y]:
            MLE_Emission['%s-->%s' % (y, x)] = data[y][x]/county    # mle for a(y,x)
        for label in y_list:
            if '%s-->%s' % (label, x) not in MLE_Emission:     # coffee: I-positive but not O, put O-->coffee as 0
                MLE_Emission['%s-->%s' % (label, x)] = 0

    return MLE_Emission


def prediction(file, parameters):
    f = open(file, 'r')
    out = open('dev.prediction', 'w')
    y_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
    for x in f.readlines():
        x = x.replace('\n', '')
        if x != '':     # x = 'coffee'
            score = 0
            label = ''
            for y in y_list:        # iterate through all labels, find the best one
                if y+'-->'+x in parameters.keys():
                    if parameters[y+'-->'+x] > score:
                        score = parameters[y+'-->'+x]
                        label = y
                else:
                    if parameters[y+'-->'+'#UNK#'] > score:
                        score = parameters[y+'-->#UNK#']
                        label = y    # if x not in parameter, label it as O; how to deal with #UNK#?
            out.write(x+' '+label+'\n')
        else:
            out.write('\n')
    f.close()
    out.close()

# prediction('dev.in', mle_emission('train_fixed'))
print(mle_emission('train_fixed'))

