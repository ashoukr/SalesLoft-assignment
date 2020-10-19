import unittest
from unittest.mock import patch
from model import SeniorityModel

import os
import pandas as pd

class TestSeniorityModel(unittest.TestCase):

    def setUp(self):
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
        self.model1.save(self.filename)
        self.model2 = SeniorityModel()
        self.model2.load(self.filename)
        
        # assuming they are getting the same data
        self.assertEqual(self.model1.predict_salesloft_team(), self.model2.predict_salesloft_team())

        os.remove(self.filename + '_model.json')
        os.remove(self.filename + '_vectorizer.json')

    def test_predict(self):
        job_titles = ['infrastructure manager', 'client', 'success', 'product development']
        preds = self.model1.predict(job_titles)
        
        self.assertEqual(len(job_titles), len(preds))
        
        
    def test_predict_salesloft_team(self):
        res = self.model1.predict_salesloft_team()
        x, y = zip(*res)
        x = list(x)
        y = list(y)
        self.assertEqual(len(set(x)), len(x))
        for elem in y:
            self.assertTrue(elem in self.job_seniority)
            
    def test_check_for_array(self):
        self.assertRaises(TypeError, "variable should be of type list or numpy array.", self.model1._check_for_array, 1.0)

            
    
        
if __name__ == '__main__':
    unittest.main()
