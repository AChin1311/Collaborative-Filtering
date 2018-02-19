import numpy as np
import os
import csv
import matplotlib.pyplot as plt
plt.close('all')




os.chdir("ml-latest-small")
os.getcwd()

def my_readcsv(filename):	
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0	
    a = []
    for row in reader:
        a.append (row)
        rownum += 1    
    ifile.close()
    return a


#movie_id = my_readcsv('movies.csv')
rating_list = my_readcsv('ratings_new.csv')


"""=== Q1 ==="""



def user_movie_Rating(rating_all):
    num_users  = 671
    num_movies = 9123
    Rating = np.zeros((num_users,num_movies))
    for i in np.arange(len(rating_all)):
        if i == 0:
            continue
        List_i = rating_all[i]
        Rating[int(List_i[0])-1,int(List_i[1])-1] = List_i[2]
    return Rating


# Users - Movies rating matrix
user_movie_R = user_movie_Rating(rating_list)
sparsity_Q1 = (len(np.nonzero(user_movie_R)[0]) / (671*9123))

print("==== Q1 ====")
print("Sparsity of rating matrix =" , sparsity_Q1)

#Sparsity of rating matrix = 0.9836635692399274




"""=== Q2 ==="""




unique, counts = np.unique(user_movie_R, return_counts=True)

counts = np.delete(counts,0)
rate_interv_freq = counts/np.sum(counts)



rate_index = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

plt.figure()
plt.bar(rate_index,rate_interv_freq, align='center',width=0.4,color='orange')
plt.title("Q2 Frequencies of rating values")




"""=== Q3 ==="""


#Q3_count = np.sum(user_movie_R,axis=0)
Q3_count = np.count_nonzero(user_movie_R,axis=0)
Q3_index_all = np.linspace(1,9123,num=9123)


Q3_index_freq = np.vstack((Q3_index_all,Q3_count))
X1 = Q3_index_freq[0]
X2 = Q3_index_freq[1]
zipped = zip(X1, X2)

Q3_index_freq_sort_list = sorted(zipped,key = lambda t: t[1])

X11,X22 = zip(*Q3_index_freq_sort_list)


# The movie indices in the order of rating received.
Q3_index_freq_out = np.flip(np.vstack((np.asarray(X11),np.asarray(X22))),axis=1)
print("==== Q3 ====")
print("The movie indices in the order of rating received")
print("is saved in Q3_index_freq_out")




plt.figure()
plt.bar(Q3_index_all,Q3_index_freq_out[1])
plt.title("Q3 Distribution of rating received by each movied (sorted)")



"""=== Q4 ==="""

Q4_count = np.count_nonzero(user_movie_R,axis=1)
Q4_index_all = np.linspace(1,671,num=671)

Q4_index_freq = np.vstack((Q4_index_all,Q4_count))

X1 = Q4_index_freq[0]
X2 = Q4_index_freq[1]

zipped = zip(X1, X2)
Q4_index_freq_sort_list = sorted(zipped,key = lambda t: t[1])

X11,X22 = zip(*Q4_index_freq_sort_list)


# The users indices in order of the "num of movies rated."
Q4_index_freq_out = np.flip(np.vstack((np.asarray(X11),np.asarray(X22))),axis=1)
print("==== Q4 ====")
print("The users indices in order of the (num of movies rated)")
print("is saved in Q4_index_freq_out")


plt.figure()
plt.bar(Q4_index_all,Q4_index_freq_out[1])
plt.title("Q4 Number of movies the users have rated (sorted)")




"""=== Q6 ==="""

print("==== Q6 ====")

# use the result in Q1   user_movie_R
# remove zeros and calculate the var


Q6_list_no0 =[]

Q6_var = np.zeros(user_movie_R.shape[1])

for i in np.arange(user_movie_R.shape[1]):
    tmp = user_movie_R[:,i]
    tmp = tmp[np.nonzero(tmp)]
    Q6_list_no0.append(tmp)
    if len(tmp)==0:
        Q6_var[i] = 10
        print(i)
    else:
        Q6_var[i] = np.var(tmp)




#Q6_var length : the number of total movies
var_index = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]


plt.figure()
plt.hist(Q6_var, bins = var_index, color="green", width=0.4 )
plt.title("Q6 Distribution of variances of rating of movies")



print("==== Q1~Q6 completed ====")


