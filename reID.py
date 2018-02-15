import csv
with open("ml-latest-small/movies.csv", "r") as csvfile:
  content = csv.reader(csvfile)


  with open("ml-latest-small/movies_new.csv", "wb") as file:
    writer = csv.writer(file)
    dic = {}
    i = 0
    for row in content:
      id, name = row[0], row[1]
      dic[id] = (i, name)
      writer.writerow([i, name])
      i += 1
  file.close()
csvfile.close()

with open("ml-latest-small/ratings.csv", "r") as csvfile:
  content = csv.reader(csvfile)
 
  with open("ml-latest-small/ratings_new.csv", "wb") as file:
    writer = csv.writer(file)
    for row in content:
      userId, movId, rating = row[0], row[1], row[2]
      name = dic[movId][1]
      id = dic[movId][0]
      print(userId, movId, rating)
      writer.writerow([userId, id, rating, name])
  file.close()
csvfile.close()  