import unittest
from unittest.mock import patch
from model import SeniorityModel

import os
import pandas as pd

class TestSeniorityModel(unittest.TestCase):

    def setUp(self):
        """ This setup is done at the beginning of all testing once since it will be used in mose of the tests.
        Initializes job_titles, job_seniority from data and uses fit() method to train model
        """
        self.model1 = SeniorityModel()
        
        data_dir = '../data'
        title_data = os.path.join(data_dir, 'title_data_for_model_fitting.csv')
        raw_data = pd.read_csv(title_data)
        self.job_titles = list(raw_data.job_title)
        self.job_seniority = list(raw_data.job_seniority)
        
        self.model1.fit(self.job_titles, self.job_seniority)
        self.filename = 'test_deleteme'
        
        
#    def tearDown(self):

    
    def test_save_load(self):
        """ Tests that predictions made by model persisted in disk is the same as the current model.
        """
        self.model1.save(self.filename)
        self.model2 = SeniorityModel()
        self.model2.load(self.filename)
        
        # assuming they are getting the same data
        self.assertEqual(self.model1.predict_salesloft_team(), self.model2.predict_salesloft_team())

        os.remove(self.filename + '_model.json')
        os.remove(self.filename + '_vectorizer.json')

    def test_predict(self):
        """ Tests that the predict function is working and giving an output per job_title passed in
        """
        job_titles = ['infrastructure manager', 'client', 'success', 'product development']
        preds = self.model1.predict(job_titles)
        
        self.assertEqual(len(job_titles), len(preds))
        
        
    def test_predict_salesloft_team(self):
        """ Tests that the ids and titles returned from api are the same lenght and that all predictions are part of original labeled data
        """
        res = self.model1.predict_salesloft_team()
        ids, titles = zip(*res)
        ids = list(ids)
        titles = list(titles)
        self.assertEqual(len(set(ids)), len(ids))
        for elem in titles:
            self.assertTrue(elem in self.job_seniority)
            
    def test_check_for_array(self):
        """ Tests that a TypeError is raised when a value is passed in to _check_for_array that is not a list or a numpy array
        """
        self.assertRaises(TypeError, "variable should be of type list or numpy array.", self.model1._check_for_array, 1.0)

            
    
        
if __name__ == '__main__':
    unittest.main()
