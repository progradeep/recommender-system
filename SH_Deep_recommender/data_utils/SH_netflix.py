# Copyright (c) 2017 NVIDIA Corporation
from os import listdir, path, makedirs
import random
import pickle
import time
import datetime
from tqdm import tqdm


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def print_stats(data):
  total_ratings = 0
  print("STATS")
  for user in data:
    total_ratings += len(data[user])
  print("Total Ratings: {}".format(total_ratings))
  print("Total User count: {}".format(len(data.keys())))

def save_data_to_file(data, filename):
  with open(filename, 'w') as out:
    for userId in data:
      for record in data[userId]:
        out.write("{}\t{}\t{}\n".format(userId, record[0], record[1]))

def create_NETFLIX_data_timesplit(all_data,
                                  train_min,
                                  train_max,
                                  test_min,
                                  test_max):
  """
  Creates time-based split of NETFLIX data into train, and (validation, test)
  :param all_data:
  :param train_min:
  :param train_max:
  :param test_min:
  :param test_max:
  :return:
  """
  train_min_ts = time.mktime(datetime.datetime.strptime(train_min,"%Y-%m-%d").timetuple())
  train_max_ts = time.mktime(datetime.datetime.strptime(train_max, "%Y-%m-%d").timetuple())
  test_min_ts = time.mktime(datetime.datetime.strptime(test_min, "%Y-%m-%d").timetuple())
  test_max_ts = time.mktime(datetime.datetime.strptime(test_max, "%Y-%m-%d").timetuple())

  training_data = dict()
  validation_data = dict()
  test_data = dict()

  train_set_items = set()

  # all_data.items() = key, values => 유저 아이디, [(아이템 아이디, 레이팅, ts), (itemId, rating, ts) ,, ]
  for userId, userRatings in all_data.items():
    time_sorted_ratings = sorted(userRatings, key=lambda x: x[2])  # sort by timestamp

    for rating_item in time_sorted_ratings:
      if rating_item[2] >= train_min_ts and rating_item[2] <= train_max_ts:
        if not userId in training_data:
          training_data[userId] = []

        training_data[userId].append(rating_item)
        train_set_items.add(rating_item[0]) # keep track of items from training set

      elif rating_item[2] >= test_min_ts and rating_item[2] <= test_max_ts:
        if not userId in training_data: # only include users seen in the training set
          continue

        p = random.random()
        if p <=0.5:
          if not userId in validation_data:
            validation_data[userId] = []
          validation_data[userId].append(rating_item)
        else:
          if not userId in test_data:
            test_data[userId] = []
          test_data[userId].append(rating_item)

  # remove items not not seen in training set
  for userId, userRatings in test_data.items():
    test_data[userId] = [rating for rating in userRatings if rating[0] in train_set_items]
  for userId, userRatings in validation_data.items():
    validation_data[userId] = [rating for rating in userRatings if rating[0] in train_set_items]

  return training_data, validation_data, test_data


def main():
  user2id_map = dict()
  item2id_map = dict()
  userId = 0
  itemId = 0
  all_data = dict()

  folder = "C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\Forks\DeepRecommender\\training_set"
  out_folder = "C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\Forks\DeepRecommender\\Netflix"
  # create necessary folders:
  for output_dir in [(out_folder + f) for f in [
    "/N3M_TRAIN", "/N3M_VALID", "/N3M_TEST", "/N6M_TRAIN",
    "/N6M_VALID", "/N6M_TEST", "/N1Y_TRAIN", "/N1Y_VALID",
    "/N1Y_TEST", "/NF_TRAIN", "/NF_VALID", "/NF_TEST"]]:
    makedirs(output_dir, exist_ok=True)

  text_files = [path.join(folder, f)
                for f in listdir(folder)
                if path.isfile(path.join(folder, f)) and ('.txt' in f)]

  for i, text_file in tqdm(enumerate(text_files)):
    with open(text_file, 'r') as f:
      print("Processing: {}".format(text_file))
      lines = f.readlines()
      item = int(lines[0][:-2]) # remove newline and :
      # print(item) 1: 에서 1
      # item이 imap에 없으면 추가하고 itemId++
      if not item in item2id_map:
        item2id_map[item] = itemId
        itemId += 1

      # 각 item 마다, 그 영화를 본 user - userId 만들고, usermap[유저이름] = 유저아이디 이렇게 저장
      for rating in tqdm(lines[1:]):
        parts = rating.strip().split(",")
        # try:
        #     print(parts, user2id_map, item2id_map )
        # except: continue

        try:
          user = int(parts[0])
        except:
          continue

        if not user in user2id_map:
          # 유저가 umap에 없으면 추가하고 userId++
          user2id_map[user] = userId
          userId += 1

        try:
          rating = float(parts[1])
        except:
          continue

        ts = int(time.mktime(datetime.datetime.strptime(parts[2],"%Y-%m-%d").timetuple()))
        if user2id_map[user] not in all_data:
          # 전체데이터[유저 아이디]에 새 리스트 생성
          all_data[user2id_map[user]] = []

        # 방금 생성한 리스트에 (아이템 아이디, rating, ts) 추가
        all_data[user2id_map[user]].append((item2id_map[item], rating, ts))


  print("STATS FOR ALL INPUT DATA")
  print_stats(all_data)
  save_obj(all_data, "all_data")

  # # Netflix full
  # (nf_train, nf_valid, nf_test) = create_NETFLIX_data_timesplit(all_data,
  #                                                               "1999-12-01",
  #                                                               "2005-11-30",
  #                                                               "2005-12-01",
  #                                                               "2005-12-31")
  # print("Netflix full train")
  # print_stats(nf_train)
  # save_data_to_file(nf_train, out_folder + "/NF_TRAIN/nf.train.txt")
  # print("Netflix full valid")
  # print_stats(nf_valid)
  # save_data_to_file(nf_valid, out_folder + "/NF_VALID/nf.valid.txt")
  # print("Netflix full test")
  # print_stats(nf_test)
  # save_data_to_file(nf_test, out_folder + "/NF_TEST/nf.test.txt")


  (n3m_train, n3m_valid, n3m_test) = create_NETFLIX_data_timesplit(all_data,
                                                                   "2005-09-01",
                                                                   "2005-11-30",
                                                                   "2005-12-01",
                                                                   "2005-12-31")
  print("Netflix 3m train")
  print_stats(n3m_train)
  save_data_to_file(n3m_train, out_folder+"/N3M_TRAIN/n3m.train.txt")
  print("Netflix 3m valid")
  print_stats(n3m_valid)
  save_data_to_file(n3m_valid, out_folder + "/N3M_VALID/n3m.valid.txt")
  print("Netflix 3m test")
  print_stats(n3m_test)
  save_data_to_file(n3m_test, out_folder + "/N3M_TEST/n3m.test.txt")

  (n6m_train, n6m_valid, n6m_test) = create_NETFLIX_data_timesplit(all_data,
                                                                   "2005-06-01",
                                                                   "2005-11-30",
                                                                   "2005-12-01",
                                                                   "2005-12-31")
  print("Netflix 6m train")
  print_stats(n6m_train)
  save_data_to_file(n6m_train, out_folder+"/N6M_TRAIN/n6m.train.txt")
  print("Netflix 6m valid")
  print_stats(n6m_valid)
  save_data_to_file(n6m_valid, out_folder + "/N6M_VALID/n6m.valid.txt")
  print("Netflix 6m test")
  print_stats(n6m_test)
  save_data_to_file(n6m_test, out_folder + "/N6M_TEST/n6m.test.txt")

  # Netflix 1 year
  (n1y_train, n1y_valid, n1y_test) = create_NETFLIX_data_timesplit(all_data,
                                                                   "2004-06-01",
                                                                   "2005-05-31",
                                                                   "2005-06-01",
                                                                   "2005-06-30")
  print("Netflix 1y train")
  print_stats(n1y_train)
  save_data_to_file(n1y_train, out_folder + "/N1Y_TRAIN/n1y.train.txt")
  print("Netflix 1y valid")
  print_stats(n1y_valid)
  save_data_to_file(n1y_valid, out_folder + "/N1Y_VALID/n1y.valid.txt")
  print("Netflix 1y test")
  print_stats(n1y_test)
  save_data_to_file(n1y_test, out_folder + "/N1Y_TEST/n1y.test.txt")

if __name__ == "__main__":
    main()

