# Machine Learning Engineer Offline Exercise

SalesLoft is looking to deploy a model to production that determines the seniority of a person based on their job title. This offline exercise will demonstrate some of your abilities related to this project.

## Requirements

- Python 3.5 or greater
- Installed dependencies (`pip install -r requirements.txt`)
- A [SalesLoft API](https://developers.salesloft.com/api.html#!/Topic/apikey) Key (the recruiter will provide this)
- Training data (`data/title_data_for_model_fitting.csv`)

## Submission

* All new dependencies added in requirements.txt
    * json for serialization / deserialization
    * requests to make request to SalesLoft API
    * decouple to take API key from .env file.

1. All 4 required functions are available in exercise/model.py along with helper functions.
    * initial setup to read data into titles and seniorities because it is used for most of the tests
    * Test for predict function
    * Test for save and load functions
    * Test for predict_salesloft_team function
    * test for _check_for_array function
2. There are 4 additional tests included in test_model.py (`python test_model.py`) along with additional helper functions.
    * Each function is documented in the code.





