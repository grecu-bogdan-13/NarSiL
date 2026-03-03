import json
import os
import nltk
import spacy
import penman
import amrlib
from tqdm import tqdm
from nltk.corpus import verbnet, wordnet as wn

class NarrativeEventExtractor:
    def __init__(self, amr_model_dir, semlink_path="./code/pb-vn2.json"):
        """
        Initializes the Narrative Engine using AMR and VerbNet/SemLink resources.
        """
        print("[Events] Initializing Narrative Event Extractor...")
        
        print("[Events] Loading Spacy...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "parser"])
        except OSError:
            print("Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "parser"])
        self.nlp.add_pipe("sentencizer")

        print("[Events] Checking NLTK resources...")
        for res in ['wordnet', 'verbnet', 'propbank']: 
            try:
                nltk.data.find(f'corpora/{res}')
            except LookupError:
                print(f"Downloading {res}...")
                nltk.download(res, quiet=True)

        try:
            if not os.path.exists(semlink_path):
                if os.path.exists("pb-vn2.json"):
                    semlink_path = "pb-vn2.json"
            
            with open(semlink_path, 'r', encoding='utf-8') as f:
                self.pb_vn_map = json.load(f)
            print(f"[Events] Loaded {len(self.pb_vn_map)} SemLink mappings.")
        except Exception as e:
            print(f"[Events] Warning: Could not load SemLink at {semlink_path}: {e}")
            self.pb_vn_map = {}
            
        self.device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        print(f"[Events] Loading AMR Model from {amr_model_dir}...")
        try:
            self.stog = amrlib.load_stog_model(model_dir=amr_model_dir, device=self.device)
        except Exception as e:
            print(f"[Events] Error loading AMR model: {e}")
            raise
        
        self.ACTION_CATEGORIES = {
            'POSSESSION': ['verb.possession'],
            'CHANGE': ['verb.change', '45.'], 
            'MOTION': ['verb.motion', '51.'], 
            'CAUSAL': ['59.', 'cause', 'force', 'prevent'],
            'ASPECTUAL': ['55.', 'begin', 'stop', 'continue']
        }

    def _classify_predicate(self, pb_roleset):
        tags = set()
        verb_lemma = pb_roleset.split('.')[0] if '.' in pb_roleset else pb_roleset.split('-')[0]

        try:
            for ss in wn.synsets(verb_lemma, pos=wn.VERB):
                lexname = ss.lexname() 
                for cat, indicators in self.ACTION_CATEGORIES.items():
                    if lexname in indicators:
                        tags.add(cat)
        except Exception:
            pass

        vn_data = self.pb_vn_map.get(pb_roleset, {})
        for vn_id in vn_data.keys():
            for cat, indicators in self.ACTION_CATEGORIES.items():
                if any(vn_id.startswith(ind) for ind in indicators if ind.endswith('.')):
                    tags.add(cat)
                
                try:
                    vn_xml = verbnet.vnclass(vn_id)
                    for pred in vn_xml.findall('.//PRED'):
                        val = pred.get('value').lower()
                        if val in indicators:
                            tags.add(cat)
                except: 
                    pass
        return list(tags)

    def _resolve_entity_name(self, graph, node_id, visited=None):
        if visited is None: visited = set()
        if node_id in visited: return "REF"
        visited.add(node_id)
        
        name_node = next((tgt for src, role, tgt in graph.triples if src == node_id and role == ':name'), None)
        if not name_node: return None

        ops = []
        for src, role, tgt in graph.triples:
            if src == name_node and role.startswith(':op'):
                try:
                    idx = int(role.replace(':op', ''))
                    ops.append((idx, tgt.strip('"')))
                except: pass
        ops.sort()
        return " ".join([o[1] for o in ops])

    def _extract_rich_events(self, amr_str):
        if not amr_str: return []
        try:
            g = penman.decode(amr_str)
        except Exception:
            return []

        instances = {t.source: t.target for t in g.instances()}
        attributes = {(t.source, t.role): t.target for t in g.attributes()}
        event_chain = []
        
        for var, concept in instances.items():
            if isinstance(concept, str) and '-' in concept and concept[-1].isdigit(): 
                tags = self._classify_predicate(concept)
                
                if not tags:
                    continue

                is_negated = attributes.get((var, ':polarity')) == "-"
                pol_prefix = "NOT_" if is_negated else ""
                
                resolved_args = {}
                for edge in g.edges(source=var):
                    if edge.role.startswith(':ARG'):
                        specific_name = self._resolve_entity_name(g, edge.target)
                        resolved_args[edge.role] = specific_name if specific_name else instances.get(edge.target, "unknown")
                
                arg_str = ", ".join([f"{r}:{v}" for r, v in resolved_args.items()])
                event_sig = f"{pol_prefix}{concept}[{'/'.join(tags)}]({arg_str})"
                event_chain.append(event_sig)
                
        return event_chain

    def process_text(self, text):
        """
        Splits text into sentences, generates AMR for each, 
        extracts events, and aggregates them.
        """
        if not text or not text.strip():
            return [], []

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return [], []

        graphs = self.stog.parse_sents(sentences, disable_progress=True)
        
        all_events = []
        for g_str in graphs:
            events = self._extract_rich_events(g_str)
            all_events.extend(events)
            
        return graphs, all_events

    def process_files(self, file_list):
        for input_path in file_list:
            if not os.path.exists(input_path):
                print(f"[Events] Skipping {input_path} (File not found)")
                continue

            if input_path.endswith('.jsonl'):
                output_path = input_path.replace('.jsonl', '_sequence.jsonl')
            else:
                output_path = input_path + "_sequence.jsonl"

            print(f"\n[Events] Processing: {os.path.basename(input_path)}")
            print(f"[Events] Output:     {os.path.basename(output_path)}")

            data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

            output_data = []
            skipped_count = 0
            processed_count = 0

            for entry in tqdm(data, desc="Extracting Actions"):
                if 'final_answer' in entry:
                    output_data.append(entry)
                    skipped_count += 1
                    continue
                
                processed_count += 1
                
                anchor_text = entry.get('anchor_text_coreferenced', "")
                _, events = self.process_text(anchor_text) 
                entry['anchor_text_events'] = events
                
                text_a = entry.get('text_a_coreferenced', "")
                _, events = self.process_text(text_a)
                entry['text_a_events'] = events
                
                text_b = entry.get('text_b_coreferenced', "")
                _, events = self.process_text(text_b)
                entry['text_b_events'] = events
                
                output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")

            print(f"[Events] Finished {os.path.basename(output_path)}")
            print(f"[Events] Processed: {processed_count}, Skipped (Resolved): {skipped_count}")