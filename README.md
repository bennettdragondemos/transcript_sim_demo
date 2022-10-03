# causeway_demo

Welcome to the Causeway transcript similarity demo.  Please follow the instructions for a pleasant installation and running experience.  Please refer to the pdf document for a description of the algorithm.

## Installation
I advise you to run this on python 3.9 or above, on a 16gb Ubuntu machine with a GPU.  You can, however, run this on a CPU machine, although it will probably take a few minutes.

### The .env file
This project uses two environment variables which are set when the project is run.  You must first copy the template file to a file called `.env`, which will allow you to set these variables:
```
cp DOT_ENV.txt .env
```
If you open this file, you will see two parameters:

* BERT_MODEL - The BERT model you are using, from HuggingFace (https://huggingface.co/).
* BERT_DEVICE - 'cuda' if you would like to use a GPU, otherwise 'cpu'.

### GPU issues
If you have not run pytorch using the GPU in your current environment, you may simply want to save yourself the hassle and run this in CPU.  In other words, only run in GPU mode if you've done it before, because it can be a pain to set up pytorch on a new machine.

### Using a virtual environment
You should definitely consider running this in a virtual environment.  To do this, type:
```
python -m venv causeway_environment
source causeway_environment/bin/activate
```
To deactivate, do:
```
deactivate
```
You could also use conda, which does help in certain circumstances when you want to configure pytorch to certain GPU models (https://pytorch.org/get-started/locally/).

### The install script
The installation can be taken care of by running:
```
./install.sh
```
This will install everything and download the necessary nltk modules.

## Running
To run the demo, first copy your XML files into a directory `bullock_transcripts`.  Now, you can run the script:
```
python transcript_sim.py bullock_transcripts
```
You will see an output of form:
```
[{'company': company_1,
  'comparedCompany': compared_company_1,
  'fileName': file_name_1,
  'similarities': [{'originalText1': original_text_1,1,
                    'originalText2': original_text_2,1,
                    'text1':text_1,1,
                    'text2':text_2,1,
                    'sim':sim_1}...]}...]
 ```
 where `comany_n` is the `n`-th company, `compared_company_n` is the company with maximum similarity to `company_n`, `file_name_n` is the file name for the XML transcript of `company_n`, `original_text_1,m` is the original text of the `m`-th line in the file for `company_n`, `original_text_2,o` is the `o`-th line in the file for `compared_company_n`, `text_1,m` is the filtered text from `original_text_1,m`, `text_2,o` is the filtered text for `original_text_2,o`, and `sim_m` is the similarity between these two texts.

```
