'''
python3 watch_file.py -p1 python3 transcript_sim.py bullock_transcripts/20220310-2613971-C.xml bullock_transcripts/20210810-2539333-C.xml -d `pwd`

python3 watch_file.py -p1 python3 transcript_sim.py bullock_transcripts/20220310-2613971-C.xml bullock_transcripts/20220113-2601172-C.xml -d `pwd`

python3 watch_file.py -p1 python3 transcript_sim.py bullock_transcripts -d `pwd`

conda install -c pytorch -c conda-forge pytorch torchvision cudatoolkit=11.6
'''
from pype3 import short_pp
from pype3 import pypeify,pypeify_namespace,p,_,_0,_1,_2,_3,_last,a,ep,db,l,lm,cl,app,m,ifta
from pype3 import tup,iff,ifp,change,select,squash,ift,d,dm,cl_has,cl_if,consec,ext
from pype3.helpers import *
from pype3.time_helpers import *
from pype3.loaders import *
from pype3.string_helpers import *
from pype3.numpy_helpers import *
from pype3.hash_helpers import *
import sys
from constants import STOP,BERT_LAYERS,MIN_SENTENCE_LENGTH,MAX_DIST,K
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize,word_tokenize
from dotenv import load_dotenv
from itertools import product
import xmltodict

#############
# CONSTANTS #
#############

load_dotenv()

BERT_MODEL=os.environ.get('BERT_MODEL')
BERT_DEVICE=os.environ.get('BERT_DEVICE')

## TOKENIZER AND MODEL

torch.cuda.empty_cache()

TOKENIZER=BertTokenizer.from_pretrained(BERT_MODEL)
MODEL=BertModel.from_pretrained(BERT_MODEL,output_hidden_states=True)

MODEL.to(BERT_DEVICE) 

###############
# LOADING XML #
###############

def load_xml_files(dirPath):

    fileNames=[os_join(dirPath,path) for path in os_listdir(dirPath)]
    contents=[]

    for fileName in fileNames:

        with open(fileName,'r') as f:

            js={'fileName':fileName,
                'contents':xmltodict.parse(f.read()),
               }

            contents.append(js)

    return contents

#####################
# BERT TOKENIZATION #
#####################

def is_acceptable_token(tk,stop=STOP):
    '''
    Helper function to determine if a token is acceptable:
    1) Has no '##'
    2) Is more than one character
    3) Is not in the stop list
    '''
    hasNoDoublePound='##' not in tk
    acceptableLen=len(tk) > 1
    notInStopList=tk not in stop

    return all([notInStopList,
                acceptableLen,
                hasNoDoublePound])


def tokenize(txt,tokenizer=TOKENIZER,bertDevice=BERT_DEVICE):
    '''
    This function tokenizes a string, returning a dictionary with:
    1) 'encoding' - The original encoding data
    2) 'indices' - The offsets of the tokens that are acceptable.
    3) 'tokens' - The tokens recovered from the tokenizer.
    '''
    encoding=tokenizer.encode_plus(txt,return_tensors='pt') 
    ids=encoding['input_ids'].squeeze().cpu().numpy()
    tokens=tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze())
    tokens=[tk.lower() for tk in tokens]
    tokens=[tk for tk in tokens if is_acceptable_token(tk)]
    indices=[index for (index,token) in enumerate(tokens)]

    encoding.to(bertDevice)

    return {'encoding':encoding,
            'indices':indices,
            'tokens':tokens,
           } 


##########################
# RUNNING THE BERT MODEL #
##########################

def model_output(encodingDct,model=MODEL):
    '''
    This accepts a dictionary output from `tokenize`, and produces a vectorization
    of the text.
    '''
    with torch.no_grad():

        return model(**encodingDct['encoding'])


def stack_sum_squeeze(states,indices):
    '''
    Takes a series of outputs of encoding blocks and:
    1) Sums them.
    2) L2-normalizes them for cosine similarity, taking the matrix
       transpose, performing the operation, and then reversing the transpose..
    3) Selects the indices of the matrix.
    4) Moves the matrix from the device to numpy.
    '''
    output=torch.stack([st for st in states])
    output=output.sum(0)
    output=output.squeeze() # (1)
    norm=torch.linalg.norm(output,dim=1)
    output=torch.transpose(output,0,1)
    output=torch.div(output,norm)
    output=torch.transpose(output,0,1) # (2)
    output=output.squeeze()
    output=output[indices] # (3)
    numpyOutput=output.cpu().numpy() # (4)

    return numpyOutput


def vectors(encodingDct,model=MODEL,layers=BERT_LAYERS):
    '''
    Computes the model output for a tokenizer, then runs `stack_sum_squeeze`
    on the hidden states.  It returns `encodingDct` with the computed matrix
    keyed by `m`.
    '''
    modelOutput=model_output(encodingDct)
    states=modelOutput['hidden_states']
    indices=encodingDct['indices']
    selectedStates=[states[l] for l in layers]
    m=stack_sum_squeeze(selectedStates,indices)
    encodingDct['m']=m

    return encodingDct


def process_text(text,originalText):
    '''
    Wrapper to tokenize, compute embeddings, and return a dictionary with 
    the original text included.
    '''
    embeddingDct=tokenize(text)
    embeddingDct=vectors(embeddingDct)
    embeddingDct['text']=text
    embeddingDct['originalText']=originalText

    return embeddingDct


##########################
# PROCESSING TRANSCRIPTS #
##########################

def preprocess(js,minLen=MIN_SENTENCE_LENGTH):
    '''
    This preprocesses the XML JSON:
    1) Recursively traversing the JSON, collecting all fields indexed by 'p'.
    2) Applying some basic filtering and preprocessing operations.
    3) Tokenizing them.
    4) Filtering the acceptable tokens, building new strings for them.
    5) Returning dictionaries containing the original text and the preprocessed
    texts.
    '''
    lines=deep_collect_fields(js,'p') # (1)
    lines=[line.lower() for line in lines \
           if is_string(line) and len(line) > minLen] # (2)
    preprocessed=[' '.join([tk for tk in word_tokenize(line) \
                            if is_acceptable_token(tk)]) \
                  for line in lines]  # (4)
    pairs=[{'text':preprocessedLine,
            'originalText':line} \
           for (line,preprocessedLine) in zip(lines,preprocessed)] # (5)

    return pairs


def process_transcript(transcriptJS):
    '''
    Build a JSON that contains the necessary fields for similarity and 
    identification of the company:
    1) Get the XML JSON and company.
    2) Preprocess the XML JSON.
    3) Run `process_text` on the results.
    4) Return a dictionary containing the results and other necessary 
    metadata.
    '''
    fileName=transcriptJS['fileName']
    xmlContents=transcriptJS['contents'] 
    companyFields=deep_collect_fields(xmlContents,'@affiliation') 
    company='' if not companyFields else companyFields[0] # (1)
    preprocessedContents=preprocess(xmlContents) # (2)
    lines=[process_text(textJS['text'],
                        textJS['originalText']) \
           for textJS in preprocessedContents] # (3)
    
    return {'lines':lines,
            'fileName':fileName,
            'company':company,
           } # (4)


##########################
# COMPUTE THE SIMILARITY #
##########################

def mat_sim(m1,m2,maxDist=MAX_DIST,k=K):
    '''
    Computes the similarity between two numpy matrices:
    1) Compute M1 M2^2.
    2) Filter distances below maxDist.
    3) Take the maximum distance for each row.
    4) Take sum, normalize by number of rows and columns.
    '''
    dists=np.dot(m1,m2.T)
    dists[dists < maxDist]=0
    mx=np.max(dists,axis=1) 
    sim=mx.sum()/(np.sqrt(m1.shape[0])*np.sqrt(m2.shape[0]))

    return sim 


def compute_sim(transcript1,transcript2,k=K):
    '''
    Function to compute the similarity between two transcripts:
    1) Extract necessary metadata.
    2) For each pair of lines in cartesian product:
       a) Filter out matrices with less than 1 row.
       b) Compute the similarities.
       c) Store `text`, `originalText`, and similarity in a JSON.
    3) For each sentence, get the maximum similar text.
    4) Get top k
    5) Compute the total similarity.
    6) Return the JSON with comparisons, similarities, and metadata. 
    '''
    company=transcript1['company']
    comparedCompany=transcript2['company']
    fileName=transcript1['fileName'] # (1)
    comparisons=[]
    textToOriginalText={}

    for (line1,line2) in product(transcript1['lines'],transcript2['lines']):

        if line1['m'].shape[0] > 1 and line2['m'].shape[0] > 1: # (a)

            textToOriginalText[line1['text']]=line1['originalText']
            comparisonJS={'text1':line1['text'],
                          'originalText1':line1['originalText'],
                          'text2':line2['text'],
                          'originalText2':line2['originalText'],
                          'sim':mat_sim(line1['m'],line2['m']), # (b)
                         } # (c)

            comparisons.append(comparisonJS) # (2)

    mergedByText1=merge_ls_dct(comparisons,'text1')
    mergedByText1={text1:sort_by_key(ls,'sim',True)[0] \
                   for (text1,ls) in mergedByText1.items()}
    maxSims=[{'text1':text1,
              'originalText1':textToOriginalText[text1],
              'text2':js['text2'],
              'originalText2':js['originalText2'],
              'sim':js['sim'],
             } for (text1,js) in mergedByText1.items()] # (3)
    maxSims=sort_by_key(maxSims,'sim',True)[:k] # (4)
    totalSim=np.sum([sm['sim'] for sm in maxSims]) # (5)
    
    return {'similarities':maxSims,
            'totalSim':totalSim,
            'fileName':fileName,
            'company':company,
            'comparedCompany':comparedCompany} # (6) 
    

########
# MAIN #
########     
      
def pipeline(dirPath):
    '''
    Main function:
    1) Load the XML files.
    2) Process the transcripts, converting content into JSON's.
    3) Run the comparisons between every non-identical transcript.
    4) For each transcript, find the most similar transcript.
    5) Sort the transcripts by their maximum similarity.
    '''
    contents=load_xml_files(dirPath) # (1)
    transcripts=[process_transcript(contentJS) for contentJS in contents] # (2)
    comparisons=[compute_sim(transcript1,transcript2) \
                 for (transcript1,transcript2) \
                 in product(transcripts,transcripts) \
                 if transcript1['fileName'] != transcript2['fileName']] # (3)
    mergedByFileName=merge_ls_dct(comparisons,'fileName')
    sortedByKey={fileName:sort_by_key(ls,'totalSim',True)[0] \
                 for (fileName,ls) in mergedByFileName.items()} # (4)
    values=dct_values(sortedByKey)
    values=sort_by_key(values,'totalSim',True) # (5)
    # values=[{'totalSim':js['totalSim'],
             # 'company':js['company'],
             # 'comparedCompany':js['comparedCompany']} for js in values]

    return values


pypeify_namespace(locals(),True)

if __name__=='__main__':

    js=pipeline(sys.argv[1])

    # short_pp(js) 
    pp.pprint(js) 
