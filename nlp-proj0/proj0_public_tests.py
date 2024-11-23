import unittest

from proj0 import CountVectorizer, calculate_accuracy, calculate_precisions, calculate_recalls

class Hw0PublicTests(unittest.TestCase):

	def test_count_vectorizer_fit(self):
		documents = [
			"it's raining cats and dogs.",
			"the cats and dogs fight all night."
		]
		vectorizer = CountVectorizer(n_gram_range=(1, 2))
		vectorizer.fit(documents, None)

		self.assertEqual(
			{
				("'s",): 0, 
				("'s", 'raining'): 1, 
				('.',): 2, 
				('all',): 3, 
				('all', 'night'): 4, 
				('and',): 5, 
				('and', 'dogs'): 6, 
				('cats',): 7, 
				('cats', 'and'): 8, 
				('dogs',): 9, 
				('dogs', '.'): 10, 
				('dogs', 'fight'): 11, 
				('fight',): 12, 
				('fight', 'all'): 13, 
				('it',): 14, 
				('it', "'s"): 15, 
				('night',): 16, 
				('night', '.'): 17, 
				('raining',): 18, 
				('raining', 'cats'): 19, 
				('the',): 20, 
				('the', 'cats'): 21
			},
			vectorizer.vocabulary_
		)


	def test_count_vectorizer_transform(self):
		train_documents = [
			"it's raining cats and dogs.",
			"the cats and dogs fight all night."
		]
		vectorizer = CountVectorizer(n_gram_range=(1, 2))
		vectorizer.fit(train_documents, None)

		# NOTE: here, we're encoding the same documents we 
		# learned our vocabulary from into bag-of-words vectors 
		test_documents = train_documents
		test_document_representations = vectorizer.transform(test_documents)

		self.assertEqual(
			[
				[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], 
				[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
			],
			test_document_representations.toarray().tolist()
		)


	def test_calculate_accuracy(self):
		y_test = [0, 0, 1, 0, 1, 1]
		y_predicted = [0, 1, 0, 0, 1, 0]
		self.assertEqual(
			3/6, 
			calculate_accuracy(y_test, y_predicted)
		)


	def test_calculate_precisions(self):
		y_test = [0, 0, 1, 0, 1, 1]
		y_predicted = [0, 1, 0, 0, 1, 0]
		self.assertEqual(
			(2/4, 1/2), 
			calculate_precisions(y_test, y_predicted)
		)


	def test_calculate_recalls(self):
		y_test = [0, 0, 1, 0, 1, 1]
		y_predicted = [0, 1, 0, 0, 1, 0]
		self.assertEqual(
			(2/3, 1/3), 
			calculate_recalls(y_test, y_predicted)
		)
