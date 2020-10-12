#coding=utf-8
import os

import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

book = "./冰与火之歌_乔治·马丁utf8.txt"
stopwords = r"D:\ANewStart\dataset\stopwords.txt"
def splitChapter():
	"""
	只拆分第一本书的73章作为数据，编码utf-8
	:return:
	"""
	with open(book, encoding="utf-8") as f:
		lines=f.readlines()
	length = len(lines)
	title = None
	f = open("标题作者.txt", "w", encoding="utf-8")
	n = 0
	for i in range(length):
		if lines[i][0]=="第":
			n+=1
			title = lines[i].strip()
			f.close()
			f = open(str(n)+title+".txt", "w", encoding="utf-8")
			f.write(lines[i])
		else:
			f.write(lines[i])
def wordslist():
	"""
	分词
	:return:
	"""
	# wordList = []
	stop_word = [line.rstrip() for line in open(stopwords, encoding="utf-8")]
	jieba.add_word(u"丹妮莉丝")
	jieba.add_word(u"提利昂")
	# fr = open("测试.txt", "a", encoding="utf-8")
	for file in os.listdir("./data")[1:-2]:
		with open("./data/"+file, "r", encoding="utf-8") as f:
			content = f.read().strip().replace("\n", "").replace(" ", "").replace("\t", "").replace("\r", "")
		seg_list = jieba.cut(content, cut_all=True)
		seg_list_after = []
		# 去停用词
		for seg in seg_list:
			if seg not in stop_word:
				seg_list_after.append(seg)
		result = " ".join(seg_list_after)
		# wordList.append(result)
		yield result
		# fr.write(result+"\n")
	# fr.close()
def titlelist():
	for file in os.listdir("./data")[1:-2]:
		yield file
if __name__ == '__main__':
	wordslist = list(wordslist())
	titlelist = list(titlelist())
	# 计算每个text中的词频
	vectorizer = CountVectorizer()
	x = vectorizer.fit_transform(wordslist)
	# print(vectorizer.get_feature_names())
	# print(x.toarray())

	# 计算tf-idf
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(x)

	# 所有文本中的关键词
	words = vectorizer.get_feature_names()

	# 所有文本中词的权重
	weights = tfidf.toarray()
	n = 5
	for (title, w) in zip(titlelist, weights):
		print(u"{}:".format(title))
		loc = np.argsort(-w) # 排序 倒排!!!
		for i in range(n):
			print(u"-{}:{} {}".format(str(i+1), words[loc[i]], w[loc[i]]))
		print("\n")
