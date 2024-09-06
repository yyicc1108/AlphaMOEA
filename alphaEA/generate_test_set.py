import os
import pandas as pd
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

zdt_instances = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
dtlz_instances = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]


def get_all_files(dir, ):
    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘results’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'results' in file:
            if file.endswith('.xlsx') or file.endswith('.csv'):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/results.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list


def init(mop_instance):
    train_data_files = get_all_files(dir='../training_data')
    for file in train_data_files:
        if mop_instance in file:
            label_data = pd.read_csv(file, header=0)

    return label_data

def generate_test_data(num_samples = 3000):

    for mop_instance in dtlz_instances:
        label_data = init(mop_instance)
        seq_len = label_data.shape[1]
        test = torch.rand(num_samples, seq_len, dtype=torch.float64)
        pd.DataFrame(test.numpy()).to_csv("../test_data/"+mop_instance+"_test.csv", index=False)

    for mop_instance in zdt_instances:
        label_data = init(mop_instance)
        seq_len = label_data.shape[1]
        test = torch.rand(num_samples, seq_len, dtype=torch.float64)
        pd.DataFrame(test.numpy()).to_csv("../test_data/"+mop_instance+"_test.csv", index=False)


if __name__ == "__main__":

    generate_test_data()





