import csv

dic = {}
dic_rate = {}
with open("ml-latest-small/ratings_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)
  for row in content:
    user, movie, rate = row[0], row[1], row[2]
    if movie not in dic:
      dic[movie] = 1
      dic_rate[movie] = rate
    else:
      dic[movie] += 1 
      dic_rate[movie] = 10
csvfile.close()

with open("ml-latest-small/ratings_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)

  with open("ml-latest-small/ratings_popular.csv", "wb") as file:
    writer = csv.writer(file)
    for row in content:
      userId, movId, rating, name = row[0], row[1], row[2], row[3]
      if dic[movId] > 2:
        writer.writerow([userId, movId, rating, name])
        
  file.close()


csvfile.close()

with open("ml-latest-small/ratings_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)

  with open("ml-latest-small/ratings_unpopular.csv", "wb") as file:
    writer = csv.writer(file)
    for row in content:
      userId, movId, rating, name = row[0], row[1], row[2], row[3]
      if dic[movId] < 3:
        writer.writerow([userId, movId, rating, name])
        
  file.close()


csvfile.close()

with open("ml-latest-small/ratings_new.csv", "r") as csvfile:
  content = csv.reader(csvfile)

  with open("ml-latest-small/ratings_var.csv", "wb") as file:
    writer = csv.writer(file)
    for row in content:
      userId, movId, rating, name = row[0], row[1], row[2], row[3]
      if dic[movId] > 4 and dic_rate[movId] == 10:
        writer.writerow([userId, movId, rating, name])
        
  file.close()


csvfile.close()

  # with open("ml-latest-small/movies_new.csv", "wb") as file:
  #   writer = csv.writer(file)
  #   dic = {}
  #   i = 0
  #   for row in content:
  #     id, name = row[0], row[1]
  #     dic[id] = (i, name)
  #     writer.writerow([i, name])
  #     i += 1
  # file.close()
