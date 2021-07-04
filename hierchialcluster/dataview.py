import pandas as pd
file=open('result1.txt','r')
span_dict = {}
dis_dict = {}
cluster = {}

for line in file:
    row = line.strip().split('\t')

    if len(row) > 1:
        word, cluster_id ,dis = row[0], row[1], row[2]
        if cluster_id not in cluster:
            cluster[cluster_id] = []
        cluster[cluster_id].append((word, float(dis), len(word.split(" "))))

for k,v in cluster.items():
    span_dict[k] = []
    dis_dict[k] = []
    for i in v:
        span_dict[k].append(i[2])
        dis_dict[k].append(i[1])
    
print(cluster)
df = pd.DataFrame(columns=('cluster_id','1','2','3','4','5','6','7', '8','9','10'))
# fw1 = open('span.csv', 'w')
fw2 = open('dis.csv', 'w')
for k, v in span_dict.items():
    sort_v = sorted(v)
    cnt = {}
    for w in sort_v:
        if w not in cnt:
            cnt[w] = 0
        else:
            cnt[w] += 1
    s = pd.Series({'cluster_id':k,
        '1':cnt[1] if 1 in cnt else 0,
        '2':cnt[2] if 2 in cnt else 0,
        '3':cnt[3] if 3 in cnt else 0,
        '4':cnt[4] if 4 in cnt else 0,
        '5':cnt[5] if 5 in cnt else 0,
        '6':cnt[6] if 6 in cnt else 0,
        '7':cnt[7] if 7 in cnt else 0,
        '8':cnt[8] if 8 in cnt else 0,
        '9':cnt[9] if 9 in cnt else 0,
        '10':cnt[10] if 10 in cnt else 0}
        )
    df = df.append(s, ignore_index=True)
    df.to_csv('spn.csv')

for k, v in dis_dict.items():
    fw2.write(str(k))
    sort_v = sorted(v)
    for w in sort_v:
        fw2.write(','+str(w))
    fw2.write('\n')




