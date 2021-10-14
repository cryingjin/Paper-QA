import json
import os
import random
from string import ascii_lowercase
import re
from tqdm import tqdm

DATA_DIR = "./data"

def combine_dataset(data_dir,output_file_name):
    outputs = []
    data_list = os.listdir(os.path.join(data_dir))
    # load
    for json_file in tqdm(data_list):
        with open(os.path.join(data_dir,json_file), "r", encoding="utf-8", errors='ignore') as f:
            data = json.load(f, strict=False)
            outputs.append(data)
    # save
    output_file = os.path.join(data_dir, str(output_file_name) + ".json")
    with open(output_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(outputs, indent=4, ensure_ascii=False) + "\n")


def split_dataset(input_json, output_dir, train_ratio, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    data_ids = [x.get('doc_id') for x in dataset]
    data_ids.sort()
    random.shuffle(data_ids)

    num_train = int(len(data_ids) * train_ratio)
    num_val = int((len(data_ids) - num_train) * 0.5)
    # num_train = len(data_ids) - num_val

    data_ids_train, data_ids_val, data_ids_test = set(data_ids[:num_train]), set(data_ids[num_train:num_train+num_val+1]), set(data_ids[num_train+num_val:])

    train_data = [x for x in dataset if x.get('doc_id') in data_ids_train]
    val_data = [x for x in dataset if x.get('doc_id') in data_ids_val]
    test_data = [x for x in dataset if x.get('doc_id') in data_ids_test]

    train = {
        'version':'paper-qa-v1',
        'data': train_data,
    }

    val = {
        'version': 'paper-qa-v1',
        'data': val_data,
    }

    test = {
        'version': 'paper-qa-v1',
        'data': test_data,
    }

    output_seed_dir = os.path.join(output_dir, f'seed{random_seed}')
    os.makedirs(output_seed_dir, exist_ok=True)
    output_train_json = os.path.join(output_seed_dir, 'train.json')
    output_val_json = os.path.join(output_seed_dir, 'val.json')
    output_test_json = os.path.join(output_seed_dir, 'test.json')

    print(f'write {output_train_json}')
    print(len(train_data))
    with open(output_train_json, 'w', encoding="utf-8", errors='ignore') as train_writer:
        json.dump(train, train_writer)

    print(f'write {output_val_json}')
    print(len(val_data))
    with open(output_val_json, 'w', encoding="utf-8", errors='ignore') as val_writer:
        json.dump(val, val_writer)

    print(f'write {output_test_json}')
    print(len(test_data))
    with open(output_test_json, 'w', encoding="utf-8", errors='ignore') as test_writer:
        json.dump(test, test_writer)


# train_data 나눠놓기
def split_train_dataset(input_json, output_dir, split_nums, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)["data"]

    data_ids = [x.get('doc_id') for x in dataset]
    data_ids.sort()
    random.shuffle(data_ids)

    split_span = int(len(data_ids)/split_nums)
    split_start = 0
    for i in range(0,split_nums):

        data_ids_train = set(data_ids[split_start:]) if i == split_nums else set(data_ids[split_start:split_start+split_span+1])
        split_data = [x for x in dataset if x.get('doc_id') in data_ids_train]
        split_start += split_span
        split_train_data = {
            'version':'paper-qa-v1',
            'data': split_data,
        }

        output_seed_dir = os.path.join(output_dir,'split_train')
        os.makedirs(output_seed_dir, exist_ok=True)
        output_train_json = os.path.join(output_seed_dir, f'train_{i}.json')

        print(f'write {output_train_json}')
        with open(output_train_json, 'w', encoding="utf-8", errors='ignore') as train_writer:
            train_writer.write(json.dumps(split_train_data, indent=4, ensure_ascii=False) + "\n")





# 사용 데이터 전체 합치기
combine_dataset(DATA_DIR, "papers_qa")

# train/validation/test 나눠서 저장
split_dataset(os.path.join(DATA_DIR,"papers_qa.json"), DATA_DIR, train_ratio=0.8, random_seed=42)

# # 불러오기
# with open(os.path.join(DATA_DIR,"seed42","train.json"), "r", encoding="utf-8") as f:
#     data = json.load(f)
#     # print(data[0])

# train 분할하기
split_train_dataset(os.path.join(DATA_DIR,"seed42","train.json"), os.path.join(DATA_DIR,"seed42"), split_nums=80, random_seed=42)

