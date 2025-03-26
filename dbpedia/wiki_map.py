# Wiki Memory: Maps Pile Wikipedia subset to DBpedia relations

import os
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
import time
import pandas as pd
from urllib.parse import quote

class WikiDBpediaMapper:
    def __init__(self, cache_dir=None):
        """Initialize the mapper with optional cache directory."""
        self.spotlight_api = "http://localhost:2222/rest/annotate"
        self.spotlight_headers = {
            "Accept": "application/json"
        }
        self.sparql_endpoint = "http://dbpedia.org/sparql"
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
    def load_pile_wikipedia(self, num_samples=1000):
        """Load the Pile Wikipedia dataset from Huggingface."""
        print("Loading Pile Wikipedia dataset...")
        dataset = load_dataset("timaeus/pile-wikipedia_en", split="train", streaming=True)
        return dataset.take(num_samples)
    
    def annotate_with_spotlight(self, text):
        """Annotate text with DBpedia Spotlight API."""
        # Check cache first
        cache_file = None
        if self.cache_dir:
            # Use hash of text to create cache filename
            text_hash = str(hash(text[:1000]))  # Use first 1000 chars for hashing
            cache_file = os.path.join(self.cache_dir, f"spotlight_{text_hash}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass  # If cache read fails, continue with API call
        
        # Limit text length for API call (Spotlight has limits)
        text_to_annotate = text #text[:5000] if len(text) > 5000 else text
        
        try:
            # Prepare request parameters
            params = {
                "text": text_to_annotate
            }
                # "confidence": 0.5,  # Confidence threshold
            
            # Call DBpedia Spotlight API
            response = requests.get(
                self.spotlight_api, 
                headers=self.spotlight_headers,
                params=params
            )
            
            if response.status_code != 200:
                print(f"Error from Spotlight API: {response.status_code} - {response.text}")
                return {"Resources": []}
                
            result = response.json()
            
            # Save to cache
            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                    
            return result
            
        except Exception as e:
            print(f"Error calling DBpedia Spotlight: {e}")
            return {"Resources": []}
    
    def extract_dbpedia_relations(self, resource_uri):
        """Query DBpedia for relations about a resource."""
        # Extract resource name from URI
        resource_name = resource_uri.split("/")[-1]
        
        # Check cache first
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"relations_{resource_name}.json")
            if os.path.exists(cache_file):
                try:
                    return pd.read_json(cache_file)
                except:
                    pass  # If cache read fails, continue with query
        
        # Query DBpedia using SPARQL
        try:
            query = f"""
            SELECT DISTINCT ?relation ?target WHERE {{
              <{resource_uri}> ?relation ?target .
              FILTER(?relation != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
              FILTER(!isLiteral(?target) || lang(?target) = '' || lang(?target) = 'en')
            }} LIMIT 30
            """
            
            headers = {
                "Accept": "application/sparql-results+json"
            }
            params = {
                "query": query,
                "format": "json"
            }
            
            response = requests.get(self.sparql_endpoint, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"Error from SPARQL endpoint: {response.status_code}")
                return pd.DataFrame()
                
            results = response.json()
            
            relations = []
            for result in results.get("results", {}).get("bindings", []):
                relation = result.get("relation", {}).get("value", "")
                target = result.get("target", {}).get("value", "")
                relations.append({"resource": resource_uri, "relation": relation, "target": target})
            
            # Save to cache
            if cache_file and relations:
                pd.DataFrame(relations).to_json(cache_file)
            
            return pd.DataFrame(relations)
            
        except Exception as e:
            print(f"Error querying DBpedia for '{resource_uri}': {e}")
            return pd.DataFrame()
    
    def process_wikipedia_sample(self, sample):
        """Process a single Wikipedia sample."""
        text = sample.get("text", "")
        if not text:
            return {"text_id": id(text), "entities": [], "dbpedia_relations": {}}
        
        # Get annotations from DBpedia Spotlight
        annotations = self.annotate_with_spotlight(text)
        
        # Extract entities and resources
        entities = []
        if "Resources" in annotations:
            for resource in annotations["Resources"]:
                entity_text = resource.get("@surfaceForm", "")
                entity_uri = resource.get("@URI", "")
                entity_types = resource.get("@types", "").split(",")
                entities.append({
                    "text": entity_text,
                    "uri": entity_uri,
                    "types": entity_types,
                    "confidence": resource.get("@similarityScore", 0)
                })
        
        # Query DBpedia for each entity
        results = {}
        for entity in entities[:5]:  # Limit to 5 entities per sample for efficiency
            time.sleep(0.5)  # Rate limiting
            entity_uri = entity["uri"]
            relations = self.extract_dbpedia_relations(entity_uri)
            if not relations.empty:
                results[entity["text"]] = {
                    "uri": entity_uri,
                    "types": entity["types"],
                    "relations": relations.to_dict(orient="records")
                }
        
        return {
            "text_id": id(text),
            "text_snippet": text[:200] + "..." if len(text) > 200 else text,
            "entities": entities,
            "dbpedia_relations": results
        }
    
    def map_dataset(self, num_samples=10, output_file="wiki_dbpedia_mapping.json"):
        """Map the Pile Wikipedia dataset to DBpedia relations."""
        dataset = self.load_pile_wikipedia(num_samples)
        
        results = []
        for sample in tqdm(dataset, desc="Processing Wikipedia samples"):
            result = self.process_wikipedia_sample(sample)
            results.append(result)
            
        # Save results to file
        pd.DataFrame(results).to_json(output_file, orient="records")
        print(f"Mapping saved to {output_file}")
        return results

def main():
    # Create cache directory for DBpedia queries
    cache_dir = "data/dbpedia_cache"
    
    # Initialize mapper
    mapper = WikiDBpediaMapper(cache_dir=cache_dir)
    
    # Map dataset (use a small number for testing)
    mapper.map_dataset(num_samples=5, output_file="data/wiki_dbpedia_mapping.json")        

if __name__ == "__main__":
    main()
