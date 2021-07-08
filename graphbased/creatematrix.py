import pandas as pd
fw = open('matrix.txt','r')
matrix_dict = {}
col_slot_name = ['O']
row_slot_name = ['O']
for line in fw:
    line = line.strip()
    r_s, c_s, p_s = line.split('\t')
    if c_s not in col_slot_name:
        col_slot_name.append(c_s)
        if ('O', c_s) not in matrix_dict:
            matrix_dict[('O',c_s)] = 0
        if (c_s,'O') not in matrix_dict:
            matrix_dict[(c_s, 'O')] = 0
    if r_s not in row_slot_name:
        row_slot_name.append(r_s)
        if ('O', r_s) not in matrix_dict:
            matrix_dict[('O',r_s)] = 0
        if (r_s,'O') not in matrix_dict:
            matrix_dict[(r_s, 'O')] = 0

    if (c_s, p_s) not in matrix_dict:
        matrix_dict[(c_s, p_s)]  = 1
    else:
        matrix_dict[(c_s, p_s)]  += 1

row_slot_name = sorted(row_slot_name)
col_slot_name_temp = sorted(row_slot_name)
for temp_slot in col_slot_name:
    if temp_slot not in col_slot_name_temp:
        col_slot_name_temp.append(temp_slot)

col_slot_name = col_slot_name_temp

df = pd.DataFrame(columns=['first']+row_slot_name)
# print(df.columns.values)
for col_s_n in col_slot_name:
    temp_d = {'first':col_s_n}
    for row_s_n in row_slot_name:
        # print(row_s_n)
        # print(col_s_n)
        # print(row_s_n)
        # print('-'*20)
        if (col_s_n, row_s_n) not in matrix_dict:
            temp_d[row_s_n] = int(0)
        else:  
            temp_d[row_s_n] = round(int(matrix_dict[(col_s_n,row_s_n)]) / 15000, 2)
            # temp_d[row_s_n] = int(matrix_dict[(col_s_n,row_s_n)]) 
    df.loc[df.shape[0]] = temp_d

df.to_csv('matrix.csv')

# for i in range(1, df.shape[0]):
#     df.loc[i][1]  /= int(sum_list[i])

# print(df.iloc[1:,1:]/sum_list)