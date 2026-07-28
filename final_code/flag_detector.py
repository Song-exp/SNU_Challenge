# ================================================================================
# SNU AI Challenge — Natural Language Processing & Orthogonal Flag Detection
# ================================================================================

import re
import spacy
from spacy.cli import download

class OrthogonalFlagDetector:
    def __init__(self):
        # Load SpaCy English model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Refined regex dictionary for extracting semantic properties (N1-N7)
        self.patterns = {
            # N1. Camera / Discourse / Editorial cues
            "N1_camera": re.compile(
                r"\b(camera|scene|zoom(s|ed|ing)?|pan(s|ned|ning)?|shot(s)?|close-up(s)?|cuts\s+to|transition(s|ed|ing)?|fade(s|ed|ing)?|screen|view\s+shift(s)?)\b", 
                re.IGNORECASE
            ),
            # N2. Aspectual phase transitions
            "N2_phase": re.compile(
                r"\b(begin|began|starts?|started|continues?|continued|finish(es|ed|ing)?|stops?|stopped|resumes?|resumed|end\s+up|proceeds?\s+to)\b", 
                re.IGNORECASE
            ),
            # N3. Script / Procedural knowledge verbs
            "N3_script": re.compile(
                r"\b(bake|mix|pour|chop|fry|serve|stir|knead|slice|adjust|secure|install|remove|assemble|disassemble|insert|attach|detach)\b", 
                re.IGNORECASE
            ),
            # N4. Referential pronouns / Coreference tracking
            "N4_referential": re.compile(
                r"\b(a|an)\s+(man|woman|boy|girl|person|player|child|dog|cat|group|gymnast|skater|rider|athlete|fighter|opponent)\b.*\b(he|she|they|his|her|their|himself|herself)\b", 
                re.IGNORECASE | re.DOTALL
            ),
            # N5. State change / Appearance anchors
            "N5_state_change": re.compile(
                r"\b(transitions?\s+from|changes?\s+(into|from|to)|switches?\s+to|now\s+wearing|different\s+(outfit|shirt|jacket|clothes))\b", 
                re.IGNORECASE
            ),
            # N6. Iterative / Cyclical operations
            "N6_iterative": re.compile(
                r"\b(again|repeatedly|multiple\s+times|several\s+times|once\s+more|back\s+and\s+forth|over\s+and\s+over)\b", 
                re.IGNORECASE
            ),
            # N7. Ordinal enumerations
            "N7_ordinal": re.compile(
                r"\b(first|initially|at\s+first|secondly|thirdly|lastly|eventually|in\s+the\s+end|ultimately)\b", 
                re.IGNORECASE
            )
        }

    def classify_syntax_spacy(self, sentence):
        """
        Classifies sentences into Type-1 (single-clause), Type-2 (complex-subordinate),
        and Type-3 (parallel-coordinated) using dependency parsing tree.
        """
        if not isinstance(sentence, str):
            return "Type-1"
        
        doc = self.nlp(sentence)
        has_subordinate_clause = False
        has_parallel_clause = False
        
        for token in doc:
            if token.dep_ in {"advcl", "ccomp"}:
                has_subordinate_clause = True
            if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
                has_parallel_clause = True
                
        if not has_subordinate_clause and not has_parallel_clause:
            return "Type-1"
        elif has_subordinate_clause:
            return "Type-2"
        else:
            return "Type-3"

    def detect_flags(self, sentence):
        """
        Returns a binary mapping vector for the 7 orthogonal flags.
        """
        if not isinstance(sentence, str):
            return {k: 0 for k in self.patterns.keys()}
        
        flags = {}
        for flag_name, regex in self.patterns.items():
            flags[flag_name] = 1 if regex.search(sentence) else 0
        return flags

    def calculate_ai_score(self, partition, flags):
        """
        Calculates the quantitative temporal ambiguity index (ai_score) using syntactic
        base scores adjusted by semantic flag modifiers.
        """
        if partition == "Type-1":
            base_score = 0.80
        elif partition == "Type-2":
            base_score = 0.40
        else:
            base_score = 0.50
            
        mod = 0.0
        if flags.get("N6_iterative", 0) == 1:
            mod += 0.30
        if flags.get("N5_state_change", 0) == 1:
            mod -= 0.30
        if flags.get("N1_camera", 0) == 1:
            mod -= 0.20
        if flags.get("N7_ordinal", 0) == 1:
            mod -= 0.20
        if flags.get("N2_phase", 0) == 1:
            mod -= 0.10
            
        final_score = max(0.0, min(1.1, base_score + mod))
        return round(final_score, 2)

    def process_sentence(self, sentence):
        """
        Runs syntactic classification, extracts flags, and computes the ambiguity score.
        """
        partition = self.classify_syntax_spacy(sentence)
        flags = self.detect_flags(sentence)
        ai_score = self.calculate_ai_score(partition, flags)
        
        result = {
            "Sentence": sentence,
            "Partition": partition,
            "ai_score": ai_score
        }
        result.update(flags)
        return result
