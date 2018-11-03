from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf1 = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier(n_neighbors = 5)
clf3 = QuadraticDiscriminantAnalysis()
clf4 = GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X,Y)

prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])
prediction4 = clf4.predict([[190, 70, 43]])

print("Decision Tree : ", prediction1)
print("KNeighborsClassifier : ", prediction2)
print("QuadraticDiscriminantAnalysis : ", prediction3)
print("GaussianNB : ", prediction4)

pred = clf1.predict(X)
pred1 = clf2.predict(X)
pred2 = clf3.predict(X)
pred3 = clf4.predict(X)

result1 = accuracy_score(Y, pred1)
result2 = accuracy_score(Y, pred2)
result3 = accuracy_score(Y, pred3)

if result1>result2 and result1>result3:
	print("KNeighborsClassifier : ")
elif result2>result1 and result2>result3:
	print("QuadraticDiscriminantAnalysis : ")
else:
	print("GaussianNB : ")