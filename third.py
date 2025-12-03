from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional



# -------------------- Data Load and Standardize --------------------
def _find_col(df, keys):
    keys = [k.lower() for k in keys]
    for col in df.columns:
        if col.strip().lower() in keys:
            return col 
    return None 



def _concat_terms(row, cols):
    vals= []
    for c in cols:
        v= str(row.get(c,'') or '').strip()
        if v and v not in vals:
            vals.append(v)
    return ' | '.join(vals)



siddha_df = pd.read_excel('NATIONAL SIDDHA MORBIDITY CODES.xls')
unani_df = pd.read_excel('NATIONAL UNANI MORBIDITY CODES.xls')
ayurveda_df = pd.read_excel('NATIONAL AYURVEDA MORBIDITY CODES (1).xls')



def standardize(df):
    code_col = _find_col(df, ['namc_code','numc_code'])
    term_cols_candidates = []
    for k in ['namc_term','numc_term','namc_term_diacritical']:
        c = _find_col(df, [k])
        if c and c not in term_cols_candidates:
            term_cols_candidates.append(c)
    short_col = _find_col(df, ['short_definition'])
    long_col = _find_col(df, ['long_definition'])
    for c in (term_cols_candidates + [code_col ,short_col,long_col]):
        if c:
            df[c] = df[c].fillna('').astype(str)
    df['NAMC_CODE'] = df[code_col].astype(str).str.strip() if code_col else ''
    df['NAMC_TERM'] = df.apply(lambda r: _concat_terms(r, term_cols_candidates), axis=1)
    df['Short_definition'] = df[short_col].fillna('').astype(str) if short_col else ''
    df['Long_definition'] = df[long_col].fillna('').astype(str) if long_col else ''
    return df[['NAMC_CODE','NAMC_TERM','Short_definition','Long_definition']]



s1 = standardize(siddha_df)
s2 = standardize(unani_df)
s3 = standardize(ayurveda_df)



namaste_data = pd.concat([s1, s2, s3], ignore_index=True)
namaste_data['NAMC_CODE'] = namaste_data['NAMC_CODE'].fillna('').astype(str).str.strip()
namaste_data = namaste_data[namaste_data['NAMC_CODE'] != '']
namaste_data = namaste_data.drop_duplicates(subset=['NAMC_CODE']).reset_index(drop=True)



namaste_data['combined_text'] = namaste_data.apply(
    lambda r: f"{r['NAMC_CODE']} {r['NAMC_TERM']} {r['Short_definition']}", axis=1
)



icd11_data = pd.read_excel('./icd_with_synonyms_and_problems.xlsx')



if 'Code' in icd11_data.columns:
    icd11_data['Code'] = icd11_data['Code'].fillna('').astype(str).str.strip()
    icd11_data = icd11_data[icd11_data['Code'] != ''].reset_index(drop=True)
else:
    icd11_data = icd11_data.iloc[0:0]



icd11_data['combined_text'] = icd11_data.apply(
    lambda r: f"{r['Code']} {r['Name']} {r['Synonyms']}", axis=1
)



for col in ['Synonyms', 'Name', 'combined_text']:
    if col in icd11_data.columns:
        icd11_data[col] = icd11_data[col].fillna('').astype(str)
    else:
        icd11_data[col] = ''



for col in ['NAMC_TERM', 'combined_text','Short_definition', 'Long_definition']:
    if col in namaste_data.columns:
        namaste_data[col] = namaste_data[col].fillna('').astype(str)
    else:
        namaste_data[col] = ''



model = SentenceTransformer('all-MiniLM-L6-v2')



namaste_embeddings = model.encode(namaste_data['Short_definition'].tolist(), convert_to_tensor=True)



####################### ------------------ Semantic Linker Class --------------------------#################################
class SemanticLinker:
    def __init__(self, icd11_synonyms, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            print(f"\n{'='*50}\nInitializing SemanticLinker\n{'='*50}")
            
            # Set device - force CPU to save memory
            self.device = torch.device('cpu')
            torch.set_grad_enabled(False)  # Disable gradient calculation
            print(f"Using device: {self.device}")
            
            # 1. First load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=None,  # Remove auth token for public models
                use_fast=True  # Use fast tokenizer if available
            )
            
            # 2. Load model with memory optimizations
            print("Loading model with memory optimizations...")
            try:
                # Clear any cached memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Load with 8-bit quantization to reduce memory usage
                self.model = AutoModel.from_pretrained(
                    model_name,
                    device_map='auto' if torch.cuda.is_available() else None,
                    torch_dtype=torch.float32,  # Use float32 for stability
                    low_cpu_mem_usage=True,     # Optimize CPU memory usage
                    use_auth_token=None         # Remove auth token for public models
                )
                
                # Move to device if not using device_map
                if not hasattr(self.model, 'hf_device_map'):
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                print("Model loaded successfully!")
                
                # Initialize attributes
                self.icd11_synonyms = icd11_synonyms
                self.icd11_embeddings = None
                self.embedding_cache = {}
                
                # Pre-compute embeddings in smaller batches
                print("Pre-computing ICD-11 embeddings in batches...")
                self._precompute_icd11_embeddings()
                print("SemanticLinker initialized successfully!")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to CPU with reduced precision...")
                # Try with a smaller model as last resort
                smaller_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
                print(f"Trying smaller model: {smaller_model}")
                self.model = AutoModel.from_pretrained(
                    smaller_model,
                    device_map=None,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to('cpu')
                self.model.eval()
                
                # Initialize attributes
                self.icd11_synonyms = icd11_synonyms
                self.icd11_embeddings = None
                self.embedding_cache = {}
                
                # Pre-compute embeddings in small batches
                print("Pre-computing ICD-11 embeddings in small batches...")
                self._precompute_icd11_embeddings(batch_size=4)
                print("SemanticLinker initialized with fallback model!")
            
        except Exception as e:
            print(f"\n{'*'*50}\nCRITICAL ERROR in SemanticLinker.__init__:\n{str(e)}\n{'*'*50}")
            import traceback
            traceback.print_exc()
            raise
    
    def _precompute_icd11_embeddings(self, batch_size=16):
        """Pre-compute and cache embeddings for all ICD-11 terms in batches"""
        if self.icd11_embeddings is None:
            print(f"Pre-computing ICD-11 embeddings in batches of {batch_size}...")
            
            # Process in batches to save memory
            all_embeddings = []
            total_batches = (len(self.icd11_synonyms) + batch_size - 1) // batch_size
            
            for i in range(0, len(self.icd11_synonyms), batch_size):
                batch = self.icd11_synonyms[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{total_batches}...")
                
                # Get embeddings for this batch
                batch_embeddings = self.embed_texts(batch)
                all_embeddings.append(batch_embeddings)
                
                # Clear memory
                del batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Combine all batch embeddings
            self.icd11_embeddings = np.vstack(all_embeddings)
            print("ICD-11 embeddings computed and cached successfully!")
    
    def get_cached_embedding(self, text):
        """Get cached embedding or compute and cache if not exists"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Compute and cache the embedding
        embedding = self.embed_texts([text])
        self.embedding_cache[text] = embedding
        return embedding
    
    def embed_texts(self, texts, batch_size=4):
        """Generate embeddings for texts in a memory-efficient way"""
        all_embeddings = []
        
        # Process in smaller batches to save memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to device and run model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    # Get model outputs
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    last_hidden = outputs.last_hidden_state
                    input_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask, 1)
                    sum_mask = torch.clamp(input_mask.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                    # If batch fails, try with smaller batch size
                    if batch_size > 1:
                        print(f"Retrying with smaller batch size: {batch_size//2}")
                        return self.embed_texts(texts, batch_size//2)
                    else:
                        raise
                        
                # Clear memory
                del inputs, outputs, last_hidden, input_mask, sum_embeddings, sum_mask, batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Combine all batch embeddings
        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])
    
    def ensure_icd11_embeddings(self):
        if self.icd11_embeddings is None:
            print("Computing ICD11 embeddings (once)...")
            self.icd11_embeddings = self.embed_texts(self.icd11_synonyms)
    
    def find_best_match(self, input_str, threshold=0.95):
        """Find the best matching ICD-11 term using semantic similarity"""
        # First check for exact match
        if input_str in self.icd11_synonyms:
            return {
                'input_term': input_str,
                'best_icd11_match': input_str,
                'similarity_score': 1.0
            }
            
        # Get or compute input embedding
        input_embedding = self.get_cached_embedding(input_str)
        
        # Calculate similarities with all ICD-11 terms
        cosine_scores = cosine_similarity(input_embedding, self.icd11_embeddings)
        best_idx = np.argmax(cosine_scores)
        best_score = cosine_scores[0, best_idx]
        if best_score >= threshold:
            return {
                'input_term': input_str,
                'best_icd11_match': self.icd11_synonyms[best_idx],
                'similarity_score': best_score
            }
        else:
            return {
                'input_term': input_str,
                'best_icd11_match': None,
                'similarity_score': best_score
            }



# Initialize icd11_synonyms for the SemanticLinker
icd11_synonyms = icd11_data['Synonyms'].fillna('').astype(str).tolist()

# Create a single global instance of SemanticLinker
print("\n" + "="*50)
print("Initializing SemanticLinker (this will only happen once)")
print("="*50 + "\n")
linker = SemanticLinker(icd11_synonyms)

def find_icd_linking(namaste_entry, linker: SemanticLinker):
    disease_term = str(namaste_entry.get('NAMC_TERM', '')).strip().lower()
    print(f"Searching ICD for term: '{disease_term}'")

    if not disease_term:
        return {'Code': '', 'Name': '', 'Synonyms': ''}

    # Basic substring search
    icd_want = icd11_data[icd11_data['Synonyms'].str.lower().str.contains(disease_term, na=False)]

    if icd_want.empty:
        icd_want = icd11_data[
            icd11_data['Name'].str.lower().str.contains(disease_term, na=False) |
            icd11_data['combined_text'].str.lower().str.contains(disease_term, na=False)
        ]

    if not icd_want.empty:
        print("Exact substring match found")
        print(icd_want.iloc[0])
        return icd_want.iloc[0].to_dict()

    print("No exact substring match. Using semantic similarity...")
    best_match = linker.find_best_match(disease_term, threshold=0.30)

    if best_match['best_icd11_match'] is not None:
        matched_row = icd11_data[icd11_data['Synonyms'] == best_match['best_icd11_match']]
        if not matched_row.empty:
            print("Semantic match found")
            print(matched_row.iloc[0])
            return matched_row.iloc[0].to_dict()

    print("No ICD match found")
    return {'Code': '', 'Name': '', 'Synonyms': ''}




# -------------------- FastAPI Setup --------------------
from fastapi import Depends
from functools import lru_cache

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a proper singleton class for SemanticLinker
class SemanticLinkerSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("\n" + "="*50)
            print("Creating SemanticLinker instance (this only happens once)")
            print("="*50 + "\n")
            cls._instance = super(SemanticLinkerSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, icd11_synonyms, model_name="xlm-roberta-base"):
        if not self._initialized:
            self.linker = SemanticLinker(icd11_synonyms, model_name)
            self._initialized = True
    
    def __getattr__(self, name):
        return getattr(self.linker, name)

def get_linker():
    # This will now always return the same instance
    if not hasattr(get_linker, '_linker'):
        get_linker._linker = SemanticLinkerSingleton(icd11_synonyms)
    return get_linker._linker

class SearchResult(BaseModel):
    NAMASTE_code: str
    NAMASTE_text: str
    closest_ICD11_code: str
    closest_ICD11_name: str
    closest_ICD11_synonyms: str 
    similarity_score: float



def build_search_result(namaste_entry, icd_entry, similarity_score):
    return SearchResult(
        NAMASTE_code = str(namaste_entry.get('NAMC_CODE', '') or ''),
        NAMASTE_text = str(namaste_entry.get('combined_text', '') or ''),
        closest_ICD11_code = str(icd_entry.get('Code', '') or ''),
        closest_ICD11_name = str(icd_entry.get('Name', '') or ''),
        closest_ICD11_synonyms = str(icd_entry.get('Synonyms', '') or ''),
        similarity_score = similarity_score,
    )




@app.get("/search", response_model=list[SearchResult])
def search_diseases(
    code: Optional[str] = Query(None, description='Search by NAMASTE Disease Code'),
    synonym: Optional[str] = Query(None, description='Search by synonym term'),
    description: Optional[str] = Query(None, description='Search by description text'),
    linker: SemanticLinker = Depends(get_linker)
):
    results = []



    if code:
        code_search = str(code).strip().lower()
        namaste_want = namaste_data[namaste_data['NAMC_CODE'].str.lower().str.contains(code_search, na=False)]
        namaste_want = namaste_want.head(10)
        for _, namaste_entry in namaste_want.iterrows():
            icd_entry = find_icd_linking(namaste_entry, linker)
            results.append(build_search_result(namaste_entry, icd_entry, similarity_score=1.0))
            if len(results) >= 10:
                break
        return results



    if synonym:
        synonym_search = str(synonym).strip().lower()
        namaste_want = namaste_data[namaste_data['NAMC_TERM'].str.lower().str.contains(synonym_search, na=False)]
        namaste_want = namaste_want.head(10)
        for _, namaste_entry in namaste_want.iterrows():
            icd_entry = find_icd_linking(namaste_entry, linker)
            results.append(build_search_result(namaste_entry, icd_entry, similarity_score=1.0))
            if len(results) >= 10:
                break
        return results



    if description:
        desc_str = str(description).strip()
        if desc_str:
            q_emb = model.encode(desc_str, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, namaste_embeddings)[0]
            top_indices = sims.topk(10).indices.cpu().numpy()
            for idx in top_indices:
                namaste_entry = namaste_data.iloc[idx]
                icd_entry = find_icd_linking(namaste_entry, linker)
                results.append(build_search_result(namaste_entry, icd_entry, similarity_score=float(sims[idx].cpu())))
                if len(results) >= 10:
                    break
        return results

    return []



# Add signal handling for graceful shutdown
import signal
import sys

def handle_exit(signum, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the FastAPI server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, handle_exit)
    
    try:
        print(f"Starting server on port {args.port}...")
        print("Press Ctrl+C to stop the server")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    finally:
        print("Server stopped")
