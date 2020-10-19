import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

import json
import requests

from decouple import config

def clean_transform_title(job_title):
    """Clean and transform job title. Remove punctuations, special characters,
    multiple spaces etc.
    """
    if not isinstance(job_title, str):
        return ''
    new_job_title = job_title.lower()
    special_characters = re.compile('[^ a-zA-Z]')
    new_job_title = re.sub(special_characters, ' ', new_job_title)
    extra_spaces = re.compile(r'\s+')
    new_job_title = re.sub(extra_spaces, ' ', new_job_title)
    
    return new_job_title


class SeniorityModel:
    """Job seniority model class. Contains attributes to fit, predict,
    save and load the job seniority model.
    """
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def _check_for_array(self, variable):
        if not isinstance(variable, (list, tuple, np.ndarray)):
            raise TypeError("variable should be of type list or numpy array.")
        return
    
    
    def _data_check(self, job_titles, job_seniorities):
        self._check_for_array(job_titles)
        self._check_for_array(job_seniorities)
        
        if len(job_titles) != len(job_seniorities):
            raise IndexError("job_titles and job_seniorities must be of the same length.")
        
        return
        
    def fit(self, job_titles, job_seniorities):
        """Fits the model to predict job seniority from job titles.
        Note that job_titles and job_seniorities must be of the same length.
        
        Parameters
        ----------
        job_titles: numpy array or list of strings representing job titles
        job_seniorities: numpy array or list of strings representing job seniorities
        """
        self._data_check(job_titles, job_seniorities)
        
        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        
        self.vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
        vectorized_data = self.vectorizer.fit_transform(cleaned_job_titles)
        self.model = LinearSVC()
        self.model.fit(vectorized_data, job_seniorities)
        
        return
    def predict(self, job_titles):
        """ Predicts job seniority from job titles
        
        Parameters
        ----------
        job_titles: numpy array or list of strings representing job titles
        """
        
        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        
        vectorized_data = self.vectorizer.transform(cleaned_job_titles)
        
        pred_seniorities = self.model.predict(vectorized_data)
#         print(type(pred_seniorities))
        return pred_seniorities
    
    def predict_salesloft_team(self):
        """ Returns ids and predicted job seniority of people on team pulled from SalesLoft api
        Note: Will always return job seniority of 25 members
        """
        
        num_people = 25
        url = "https://api.salesloft.com/v2/people.json?per_page=" + str(num_people)
        key = config('KEY')
        payload = {}
        headers = {
          'Authorization': 'Bearer ' + key
        }

        response = requests.request("GET", url, headers=headers, data = payload)

        response = response.text.encode('utf8')
        response = json.loads(response)

        ids = [x['id'] for x in response['data']]
        titles = [x['title'] for x in response['data']]
        
        pred_seniorities = self.predict(titles)
        
        res = [(idx, pred_sen) for idx, pred_sen in zip(ids, pred_seniorities)]
        return res
    
    def save(self, filename):
        """ Saves model and vectorizer members of class in json format
        model saved in filename_model.json
        vectorizer saved in filename_vectorizer.json
        Parameters
        ----------
        filename: Used to save model and vectorizer json files
        """
        json_vectorizer = self._serialize_vectorizer()
        json_model = self._serialize_model()
        
        with open(filename + '_vectorizer.json', 'w') as file:
            file.write(json_vectorizer)
        with open(filename + '_model.json', 'w') as file:
            file.write(json_model)
        
    def load(self, filename):
        """ Retrieves vectroizer and model using filename.
        
        Parameters
        ----------
        filename: used to retrieve model and vectorizer json files
        """
        with open(filename + '_vectorizer.json', 'r') as file:
            json_vectorizer = json.load(file)
        with open(filename + '_model.json', 'r') as file:
            json_model = json.load(file)
        
        self._deserialize_vectorizer(json_vectorizer)
        self._deserialize_model(json_model)
        
        
        
    
    def _serialize_vectorizer(self):
        """ Returns json of all vectorizer marameters
        """
        vectorizer_dict = self.vectorizer.__dict__.copy()
        
        vectorizer_dict['stop_words_'] = list(vectorizer_dict['stop_words_'])

        vectorizer_dict.pop('dtype', None)
        # proper format vocabulary_ dict vals
        for key in vectorizer_dict['vocabulary_'].keys():
            if isinstance(vectorizer_dict['vocabulary_'][key], np.int64):
                vectorizer_dict['vocabulary_'][key] = vectorizer_dict['vocabulary_'][key].item()
                
        return json.dumps(vectorizer_dict, indent = 4)
    
    def _deserialize_vectorizer(self, saved_dict):
        """ Creates new vectorizer from saved_dict and saves it as a class member

        Parameters
        ----------
        saved_dict: Used to restore all parameters to new vectorizer instance
        """
        self.vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
        saved_dict['stop_words_'] = set(saved_dict['stop_words_'])
        for key in saved_dict['vocabulary_'].keys():
            saved_dict['vocabulary_'][key] = np.int64(saved_dict['vocabulary_'][key])
            
        for key in saved_dict.keys():
            self.vectorizer.__dict__[key] = saved_dict[key]
        return
        
    def _serialize_model(self):
        model_dict = self.model.__dict__.copy()
        
        model_dict['n_iter_'] = model_dict['n_iter_'].item()
        model_dict['coef_'] = model_dict['coef_'].tolist()
        model_dict['classes_'] = model_dict['classes_'].tolist()
        model_dict['intercept_'] = model_dict['intercept_'].tolist()
#         print([type(model_dict[x]) for x in model_dict.keys()])
        return json.dumps(model_dict, indent = 4)
    
    def _deserialize_model(self, saved_dict):
        """ Creates new model from saved_dict and saves it as a class member

        Parameters
        ----------
        saved_dict: Used to restore all parameters to new model instance
        """
        saved_dict['coef_'] = np.array(saved_dict['coef_'])
        saved_dict['classes_'] = np.array(saved_dict['classes_'])
        saved_dict['intercept_'] = np.array(saved_dict['intercept_'])
        saved_dict['n_iter_'] = np.int32(saved_dict['n_iter_'])
        
        self.model = LinearSVC()
        
        for key in saved_dict.keys():
            self.model.__dict__[key] = saved_dict[key]
        return
    
    
