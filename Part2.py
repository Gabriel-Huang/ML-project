def fix_input(input):
    dic = {}
    for i in input:
        x = i.split()[0]
        if x not in dic:
            dic[x] = 1
        else:
            dic[x] += 1
    output = []
    for j in input:
        x = j.split()[0]
        if dic[x] < 3:
            j = j.replace(x, '#UNK#')
            output.append(j)
        else:
            output.append(j)
    return output


def mle_emission(file):
    with open(file) as f:
        content = [line.strip() for line in f if line.strip()]
    content = fix_input(content)
    data = {}
    for i in content:
        x = i.split()[0]
        y = i.split()[1]
        if x not in data:
            newx = {}
            newx[y] = 1
            data[x] = newx
        else:
            if y in data[x]:
                data[x][y] += 1
            else:
                data[x][y] = 1
    MLE_Emission = {}
    y_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
    for x in data:
        countx = 0
        for y in data[x]:
            countx += data[x][y]
        for y in data[x]:
            MLE_Emission['%s-->%s' % (y, x)] = data[x][y]/countx
        for label in y_list:
            if '%s-->%s' % (label, x) not in MLE_Emission:
                MLE_Emission['%s-->%s' % (label, x)] = 0

    return MLE_Emission


def prediction(file, parameters):
    f = open(file, 'r')
    out = open('dev.prediction', 'w')
    y_list = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative',
              'B-neutral', 'I-neutral']
    for x in f.readlines():
        x = x.replace('\n', '')
        if x != '':
            score = 0
            label = ''
            for y in y_list:
                if y+'-->'+x in parameters.keys():
                    if parameters[y+'-->'+x] > score:
                        score = parameters[y+'-->'+x]
                        label = y
                else:
                    label = 'O'
            out.write(x+' '+label+'\n')
        else:
            out.write('\n')
    f.close()
    out.close()

prediction('dev.in', mle_emission('train'))
